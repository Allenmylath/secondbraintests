import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional

import httpx
import pybase64
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Import the Master Realtor prompt template
from realtor_prompt import MASTER_REALTOR_PROMPT

# Load environment variables from .env file
load_dotenv()


class PropertyImageProcessor:
    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_model: str = None,
        rate_limit: int = 100,
        max_retries: int = 3,
        max_size_mb: float = 50.0,
        max_concurrent: int = 5,
        output_file: str = "property_analysis_results.json",
    ):
        # Load from environment variables or use provided values
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv(
            "GEMINI_MODEL", "gemini-2.0-flash"
        )
        self.output_file = output_file

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY must be provided either as parameter or in .env file"
            )

        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.max_size_mb = max_size_mb
        self.max_concurrent = max_concurrent

        # Configure Gemini client
        self.client = genai.Client(api_key=self.gemini_api_key)

        # Rate limiting
        self.last_request_times = []

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Timing tracking
        self.total_start_time = None
        self.property_count = 0
        self.completed_count = 0
        self.successful_analyses = 0
        self.failed_analyses = 0
        self.total_images_processed = 0
        self.rate_limit_errors = 0
        self.source_data_url = None

        # Results storage
        self.results = {
            "analysis_metadata": {},
            "properties": [],
            "processing_summary": {},
        }

        print(f"✅ Initialized with Gemini model: {self.gemini_model}")
        print(f"📁 Output will be saved to: {self.output_file}")

    def load_json_from_url(self, url: str) -> Dict:
        """Load and parse JSON data from URL."""
        try:
            print(f"📥 Fetching data from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            print(f"✅ Successfully loaded JSON data")
            return data

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to fetch data from URL: {e}")
            return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return {}

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse the JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {e}")
            return {}

    def create_prompt_with_details(self, property_details: Dict) -> str:
        """Create a comprehensive Master Realtor Assistant prompt with property details."""

        # Extract property details for placeholders
        address = property_details.get("address", "Not specified")

        # Format price
        price_info = "Not specified"
        if property_details.get("price"):
            currency = property_details.get("currency", "")
            price_formatted = (
                f"${property_details['price']:,}"
                if isinstance(property_details["price"], (int, float))
                else property_details["price"]
            )
            price_info = f"{price_formatted} {currency}".strip()

        description = property_details.get("description", "Not provided")

        # Build structured details
        structured_details = []
        if property_details.get("bedrooms"):
            structured_details.append(f"Bedrooms: {property_details['bedrooms']}")
        if property_details.get("bathrooms"):
            structured_details.append(f"Bathrooms: {property_details['bathrooms']}")
        if property_details.get("property_type"):
            structured_details.append(f"Type: {property_details['property_type']}")

        structured_details_text = (
            " | ".join(structured_details) if structured_details else "Not specified"
        )

        # Fill in the placeholders in the Master Realtor prompt
        full_prompt = MASTER_REALTOR_PROMPT.format(
            property_address=address,
            property_price=price_info,
            property_description=description,
            property_details=structured_details_text,
        )

        return full_prompt

    def extract_property_details(self, property_data: Dict) -> Dict:
        """Extract key property details from JSON data."""
        details = {}

        # Property address
        if "property_address" in property_data:
            details["address"] = property_data["property_address"]
        elif "address" in property_data and isinstance(property_data["address"], dict):
            details["address"] = property_data["address"].get("streetAddress", "")

        # Listed price
        if "price" in property_data:
            details["price"] = property_data["price"]
        if "currency" in property_data:
            details["currency"] = property_data["currency"]

        # MLS Description
        if "description" in property_data:
            details["description"] = property_data["description"]
        elif "meta_description" in property_data:
            details["description"] = property_data["meta_description"]

        # Structured details
        if "bedrooms" in property_data:
            details["bedrooms"] = property_data["bedrooms"]
        if "bathrooms" in property_data:
            details["bathrooms"] = property_data["bathrooms"]
        if "property_type" in property_data:
            details["property_type"] = property_data["property_type"]

        return details

    def extract_properties_and_images(self, data: Dict) -> Dict[str, Dict]:
        """Extract property URLs, their JPEG images, and property details from the JSON data."""
        properties_dict = {}

        # Check if data has 'properties' key (like in paste.txt format)
        if "properties" in data:
            properties = data["properties"]
        else:
            # Assume the data itself is the properties dict
            properties = data

        for property_url, property_data in properties.items():
            if isinstance(property_data, dict) and "images" in property_data:
                # Filter only JPEG images
                jpeg_images = [
                    img_url
                    for img_url in property_data["images"]
                    if img_url.lower().endswith((".jpeg", ".jpg"))
                ]

                if jpeg_images:
                    # Extract property details
                    property_details = self.extract_property_details(property_data)

                    properties_dict[property_url] = {
                        "images": jpeg_images,
                        "details": property_details,
                    }
                    self.logger.info(
                        f"Found {len(jpeg_images)} JPEG images for {property_url}"
                    )

        return properties_dict

    async def process_properties_from_url(self, url: str):
        """Main method to process properties from JSON URL and save results."""
        print(f"Starting property processing from URL...")
        self.source_data_url = url
        data = self.load_json_from_url(url)

        if not data:
            print("❌ No data found from URL")
            return

        await self._process_data(data)

    async def process_properties_from_file(self, file_path: str):
        """Main method to process properties from JSON file and save results."""
        print(f"Loading JSON file: {file_path}")
        self.source_data_url = file_path
        data = self.load_json_file(file_path)

        if not data:
            print("❌ No data found in file")
            return

        await self._process_data(data)

    async def _process_data(self, data: Dict):
        """Process the loaded data."""
        self.total_start_time = time.time()

        # Initialize metadata
        self._init_analysis_metadata()

        print(f"🔍 Extracting properties and JPEG images...")
        properties_dict = self.extract_properties_and_images(data)

        if not properties_dict:
            print("❌ No properties with JPEG images found")
            return

        self.property_count = len(properties_dict)
        self.completed_count = 0

        print(f"🏠 Found {self.property_count} properties to process")
        print(f"⏰ Starting processing at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Process each property
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(client, property_url, property_info):
            property_start_time = time.time()
            async with semaphore:
                description, error_msg = await self._process_single_property(
                    client,
                    property_url,
                    property_info["images"],
                    property_info["details"],
                )
                property_end_time = time.time()
                property_duration = property_end_time - property_start_time

                return {
                    "url": property_url,
                    "description": description,
                    "property_details": property_info["details"],
                    "processing_time": property_duration,
                    "images_processed": len(property_info["images"]),
                    "error_message": error_msg,
                }

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                process_with_semaphore(client, property_url, property_info)
                for property_url, property_info in properties_dict.items()
            ]

            # Process and store results as they complete
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    self.completed_count += 1
                    self._store_result(result, properties_dict[result["url"]]["images"])
                    self._print_progress_simple(result)
                except Exception as e:
                    self.logger.error(f"Error processing property: {e}")
                    self.failed_analyses += 1

        # Finalize and save results
        self._finalize_results()
        self._save_results()
        self._print_final_summary()

    def _init_analysis_metadata(self):
        """Initialize analysis metadata."""
        self.results["analysis_metadata"] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "gemini_model_used": self.gemini_model,
            "source_data_url": self.source_data_url,
        }

    def _store_result(self, result: Dict, image_urls: List[str]):
        """Store processing result in JSON structure."""
        # Determine status and update counters
        if result["description"] and not result["error_message"]:
            status = "success"
            self.successful_analyses += 1
        else:
            status = "failed"
            self.failed_analyses += 1
            if "429" in str(result["error_message"]):
                self.rate_limit_errors += 1

        self.total_images_processed += result["images_processed"]

        # Create property entry
        property_entry = {
            "property_url": result["url"],
            "property_details": {
                "address": result["property_details"].get("address"),
                "listed_price": result["property_details"].get("price"),
                "currency": result["property_details"].get("currency"),
                "bedrooms": result["property_details"].get("bedrooms"),
                "bathrooms": result["property_details"].get("bathrooms"),
                "property_type": result["property_details"].get("property_type"),
                "mls_description": result["property_details"].get("description"),
            },
            "processing_info": {
                "processing_time_seconds": round(result["processing_time"], 1),
                "images_processed": result["images_processed"],
                "images_analyzed": image_urls,
                "status": status,
                "error_message": result["error_message"],
            },
            "ai_analysis_raw": result["description"],
        }

        self.results["properties"].append(property_entry)

    def _finalize_results(self):
        """Finalize results with summary information."""
        total_time = time.time() - self.total_start_time

        # Update metadata
        self.results["analysis_metadata"].update(
            {
                "total_properties_processed": self.completed_count,
                "processing_time_seconds": round(total_time, 1),
            }
        )

        # Add processing summary
        self.results["processing_summary"] = {
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "total_images_processed": self.total_images_processed,
            "rate_limit_errors": self.rate_limit_errors,
        }

    def _save_results(self):
        """Save results to JSON file."""
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def _print_progress_simple(self, result: Dict):
        """Print simple progress update."""
        status_emoji = (
            "✅" if result["description"] and not result["error_message"] else "❌"
        )
        print(
            f"{status_emoji} {self.completed_count}/{self.property_count} | "
            f"{self._format_time(result['processing_time'])} | "
            f"{result['images_processed']} images"
        )

    def _print_progress(self):
        """Print current progress."""
        elapsed_time = time.time() - self.total_start_time
        avg_time_per_property = (
            elapsed_time / self.completed_count if self.completed_count > 0 else 0
        )
        remaining_properties = self.property_count - self.completed_count
        estimated_remaining_time = avg_time_per_property * remaining_properties

        print(f"📊 Progress: {self.completed_count}/{self.property_count} completed")
        print(
            f"⏱️  Elapsed: {self._format_time(elapsed_time)} | "
            f"Avg per property: {self._format_time(avg_time_per_property)} | "
            f"ETA: {self._format_time(estimated_remaining_time)}"
        )
        print("-" * 40)

    def _print_final_summary(self):
        """Print final processing summary."""
        total_time = time.time() - self.total_start_time

        print("\n" + "=" * 80)
        print("🎉 PROCESSING COMPLETE!")
        print(
            f"📈 Total Properties Processed: {self.completed_count}/{self.property_count}"
        )
        print(f"✅ Successful Analyses: {self.successful_analyses}")
        print(f"❌ Failed Analyses: {self.failed_analyses}")
        print(f"🖼️  Total Images Processed: {self.total_images_processed}")
        print(f"⚠️  Rate Limit Errors: {self.rate_limit_errors}")
        print(f"⏰ Total Processing Time: {self._format_time(total_time)}")
        print(f"💾 Results saved to: {self.output_file}")
        print("=" * 80)

    def _format_time(self, seconds: float) -> str:
        """Format time in a readable way."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
        else:
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"

    def _print_result(self, result: Dict):
        """Print the result in a formatted way with property details."""
        # This method is no longer used but kept for compatibility
        pass

    async def _process_single_property(
        self,
        client: httpx.AsyncClient,
        property_url: str,
        image_urls: List[str],
        property_details: Dict,
    ) -> tuple[Optional[str], Optional[str]]:
        """Process a single property's images with property details in prompt."""
        try:
            # Download and encode images within size limit
            images_data = await self._download_and_encode_images(client, image_urls)

            if not images_data:
                self.logger.warning(f"No images processed for {property_url}")
                return None, "No images could be processed"

            # Call Gemini API with property details in prompt
            description = await self._call_gemini_api(images_data, property_details)
            self.logger.info(f"✅ Processed {property_url}: {len(images_data)} images")

            return description, None

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to process {property_url}: {error_msg}")
            return None, error_msg

    async def _download_and_encode_images(
        self, client: httpx.AsyncClient, image_urls: List[str]
    ) -> List[str]:
        """Download images and encode to base64, respecting size limit."""
        images_data = []
        total_size_bytes = 0
        max_size_bytes = self.max_size_mb * 1024 * 1024

        for image_url in image_urls:
            try:
                response = await client.get(image_url)
                response.raise_for_status()

                # Encode to base64
                image_base64 = pybase64.b64encode(response.content).decode("utf-8")
                image_size_bytes = len(image_base64.encode("utf-8"))

                # Check size limit
                if total_size_bytes + image_size_bytes > max_size_bytes:
                    self.logger.info(
                        f"Size limit reached. Processed {len(images_data)} images"
                    )
                    break

                images_data.append(image_base64)
                total_size_bytes += image_size_bytes

            except Exception as e:
                self.logger.warning(f"Failed to download {image_url}: {e}")
                continue

        return images_data

    async def _call_gemini_api(
        self, images_base64: List[str], property_details: Dict
    ) -> str:
        """Call Gemini API with retry logic and rate limiting."""
        await self._enforce_rate_limit()

        for attempt in range(self.max_retries + 1):
            try:
                # Create prompt with property details
                full_prompt = self.create_prompt_with_details(property_details)

                # Prepare content
                contents = [full_prompt]

                # Add all images as inline data
                for img_b64 in images_base64:
                    contents.append(
                        types.Part.from_bytes(
                            data=pybase64.b64decode(img_b64), mime_type="image/jpeg"
                        )
                    )

                # Generate content using Gemini API
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.gemini_model,
                    contents=contents,
                )

                return response.text

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise e

    async def _enforce_rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()

        # Remove old timestamps
        self.last_request_times = [
            t for t in self.last_request_times if current_time - t < 60
        ]

        # Wait if at limit
        if len(self.last_request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.last_request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.last_request_times.append(current_time)


# Usage example
async def main():
    processor = PropertyImageProcessor()

    # Process from S3 URL
    s3_url = "https://secondbrainoldnewurls.s3.us-east-1.amazonaws.com/comparison_results_20250630_020123.json"
    await processor.process_properties_from_url(s3_url)

    # Or process from local file (alternative)
    # await processor.process_properties_from_file('paste.txt')


if __name__ == "__main__":
    asyncio.run(main())
