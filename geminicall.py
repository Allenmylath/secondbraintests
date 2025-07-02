import asyncio
import json
import logging
import os
import time
from typing import Dict, List, AsyncGenerator, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading

import httpx
import pybase64
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm

# Import the Master Realtor prompt template
from realtor_prompt import MASTER_REALTOR_PROMPT

# Load environment variables from .env file
load_dotenv()


class PropertyImageProcessor:
    def __init__(
        self,
        gemini_api_key: str = None,
        gemini_model: str = None,
        rate_limit: int = 200,
        max_retries: int = 3,
        max_size_mb: float = 50.0,
        max_concurrent: int = 100,
        output_file: str = "property_analysis_results.json",
        log_file: str = None,
    ):
        # Load from environment variables or use provided values
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.0-pro")
        self.output_file = output_file
        self.log_file = (
            log_file or f"property_analysis_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )

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

        # Thread pool for Gemini API calls
        self.gemini_executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="gemini_api"
        )

        # Rate limiting (200 per minute = ~3.33 per second) with thread safety
        self.last_request_times = []
        self._rate_limit_lock = asyncio.Lock()

        # Setup detailed file logging
        self._setup_logging()

        # Progress tracking
        self.images_processed = 0
        self.api_responses = 0
        self.total_properties = 0
        self.total_images = 0

        print(f"✅ Initialized with Gemini model: {self.gemini_model}")
        print(f"📁 Output will be saved to: {self.output_file}")
        print(f"📝 Detailed logs will be saved to: {self.log_file}")

    def _setup_logging(self):
        """Setup detailed file logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.log_file, mode="w", encoding="utf-8"),
                logging.StreamHandler(),  # This will be suppressed during tqdm
            ],
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("PropertyImageProcessor initialized")

    def extract_properties_from_json(self, data: Dict) -> Dict[str, Dict]:
        """Extract property URLs, their JPEG images, and property details from JSON data."""
        self.logger.info("Starting property extraction from JSON data")
        properties_dict = {}

        # Check if data has 'properties' key
        if "properties" in data:
            properties = data["properties"]
        else:
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
                    property_details = self._extract_property_details(property_data)

                    properties_dict[property_url] = {
                        "images": jpeg_images,
                        "details": property_details,
                    }
                    self.logger.info(
                        f"Found {len(jpeg_images)} JPEG images for {property_url}"
                    )

        self.total_properties = len(properties_dict)
        self.total_images = sum(
            len(prop["images"]) for prop in properties_dict.values()
        )

        self.logger.info(
            f"Extracted {self.total_properties} properties with {self.total_images} total images"
        )
        return properties_dict

    def _extract_property_details(self, property_data: Dict) -> Dict:
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

    def _create_prompt_with_details(self, property_details: Dict) -> str:
        """Create comprehensive Master Realtor Assistant prompt with property details."""
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

    async def process_images(
        self, client: httpx.AsyncClient, image_urls: List[str]
    ) -> AsyncGenerator[Tuple[str, int], None]:
        """Generator that downloads and encodes images one by one."""
        for image_url in image_urls:
            try:
                self.logger.debug(f"Downloading image: {image_url}")

                # Download image with retries
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.get(image_url, timeout=30.0)
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if attempt < self.max_retries:
                            wait_time = 2**attempt
                            self.logger.warning(
                                f"Download attempt {attempt + 1} failed for {image_url}, retrying in {wait_time}s: {e}"
                            )
                            await asyncio.sleep(wait_time)
                        else:
                            raise e

                # Encode to base64
                image_base64 = pybase64.b64encode(response.content).decode("utf-8")
                image_size_bytes = len(image_base64.encode("utf-8"))

                self.logger.debug(
                    f"Successfully encoded image: {image_url} ({image_size_bytes} bytes)"
                )
                yield image_base64, image_size_bytes

            except Exception as e:
                self.logger.error(f"Failed to process image {image_url}: {e}")
                continue

    async def process_single_property(
        self,
        client: httpx.AsyncClient,
        property_url: str,
        property_data: Dict,
        pbar_images: tqdm,
        pbar_responses: tqdm,
    ) -> Dict:
        """Process a single property with memory-efficient image handling."""
        property_start_time = time.time()

        try:
            self.logger.info(f"Starting processing of property: {property_url}")

            images_batch = []
            batch_size_bytes = 0
            max_size_bytes = self.max_size_mb * 1024 * 1024
            images_processed_count = 0

            # Generator: Process images one by one
            async for encoded_image, size_bytes in self.process_images(
                client, property_data["images"]
            ):
                # Check if adding this image would exceed size limit
                if batch_size_bytes + size_bytes > max_size_bytes and images_batch:
                    self.logger.warning(
                        f"Size limit reached for {property_url}. Processing {len(images_batch)} images"
                    )
                    break

                images_batch.append(encoded_image)
                batch_size_bytes += size_bytes
                images_processed_count += 1

                # Update progress bar
                pbar_images.update(1)

            if not images_batch:
                error_msg = "No images could be processed"
                self.logger.error(f"Failed to process {property_url}: {error_msg}")
                return self._create_error_result(
                    property_url, property_data, error_msg, property_start_time
                )

            # Auto-trigger API call when batch ready
            self.logger.info(
                f"Sending {len(images_batch)} images to Gemini API for {property_url}"
            )
            description = await self._call_gemini_api(
                images_batch, property_data["details"]
            )

            # Update API response progress bar
            pbar_responses.update(1)

            property_end_time = time.time()
            processing_time = property_end_time - property_start_time

            self.logger.info(
                f"Successfully processed {property_url} in {processing_time:.1f}s"
            )

            result = {
                "property_url": property_url,
                "property_details": self._format_property_details(
                    property_data["details"]
                ),
                "processing_info": {
                    "processing_time_seconds": round(processing_time, 1),
                    "images_processed": images_processed_count,
                    "images_analyzed": property_data["images"][:images_processed_count],
                    "status": "success",
                    "error_message": None,
                },
                "ai_analysis_raw": description,
            }

            # Clear memory immediately
            del images_batch
            return result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Failed to process {property_url}: {error_msg}")
            return self._create_error_result(
                property_url, property_data, error_msg, property_start_time
            )

    def _create_error_result(
        self, property_url: str, property_data: Dict, error_msg: str, start_time: float
    ) -> Dict:
        """Create error result structure."""
        processing_time = time.time() - start_time
        return {
            "property_url": property_url,
            "property_details": self._format_property_details(property_data["details"]),
            "processing_info": {
                "processing_time_seconds": round(processing_time, 1),
                "images_processed": 0,
                "images_analyzed": [],
                "status": "failed",
                "error_message": error_msg,
            },
            "ai_analysis_raw": None,
        }

    def _format_property_details(self, details: Dict) -> Dict:
        """Format property details for output."""
        return {
            "address": details.get("address"),
            "listed_price": details.get("price"),
            "currency": details.get("currency"),
            "bedrooms": details.get("bedrooms"),
            "bathrooms": details.get("bathrooms"),
            "property_type": details.get("property_type"),
            "mls_description": details.get("description"),
        }

    async def process_all_properties(
        self, properties_dict: Dict[str, Dict]
    ) -> AsyncGenerator[Dict, None]:
        """Generator that processes all properties with true concurrent processing."""
        # Setup progress bars
        pbar_images = tqdm(total=self.total_images, desc="Images Processed", position=0)
        pbar_responses = tqdm(
            total=self.total_properties, desc="API Responses", position=1
        )

        # Use a semaphore for concurrency control but allow true parallelism
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_with_semaphore(client, property_url, property_data):
            async with semaphore:
                return await self.process_single_property(
                    client, property_url, property_data, pbar_images, pbar_responses
                )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create all tasks at once for true concurrent processing
                tasks = [
                    asyncio.create_task(
                        process_with_semaphore(client, property_url, property_data)
                    )
                    for property_url, property_data in properties_dict.items()
                ]

                # Yield results as they complete (not in order!)
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        yield result
                        # Memory is cleared automatically in process_single_property
                    except Exception as e:
                        self.logger.error(f"Error processing property: {e}")

        finally:
            pbar_images.close()
            pbar_responses.close()

    def __del__(self):
        """Cleanup thread pool executor."""
        if hasattr(self, "gemini_executor"):
            self.gemini_executor.shutdown(wait=True)

    async def _call_gemini_api(
        self, images_base64: List[str], property_details: Dict
    ) -> str:
        """Call Gemini API with retry logic and rate limiting - fully async and non-blocking."""
        await self._enforce_rate_limit()

        for attempt in range(self.max_retries + 1):
            try:
                # Create prompt with property details
                full_prompt = self._create_prompt_with_details(property_details)

                # Prepare content
                contents = [full_prompt]

                # Add all images as inline data
                for img_b64 in images_base64:
                    contents.append(
                        types.Part.from_bytes(
                            data=pybase64.b64decode(img_b64), mime_type="image/jpeg"
                        )
                    )

                self.logger.debug(
                    f"Sending request to Gemini API (attempt {attempt + 1}) - Thread: {threading.current_thread().name}"
                )

                # Use dedicated thread pool executor for true async, non-blocking calls
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    self.gemini_executor,
                    self._sync_gemini_call,
                    contents,
                )

                self.logger.debug(
                    f"Successfully received response from Gemini API - Thread: {threading.current_thread().name}"
                )
                return response.text

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"API attempt {attempt + 1} failed, retrying in {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"All API attempts failed: {e}")
                    raise e

    def _sync_gemini_call(self, contents):
        """Synchronous Gemini API call to be run in thread pool."""
        try:
            return self.client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
            )
        except Exception as e:
            # Log the thread info for debugging
            self.logger.debug(
                f"Gemini API call failed in thread {threading.current_thread().name}: {e}"
            )
            raise e

    async def _enforce_rate_limit(self):
        """Rate limiting: 200 requests per minute (3.33 per second) - thread-safe and non-blocking."""
        async with self._rate_limit_lock:
            current_time = time.time()

            # Remove old timestamps (older than 1 minute)
            self.last_request_times = [
                t for t in self.last_request_times if current_time - t < 60
            ]

            # If we're at the rate limit, find the minimum wait time
            if len(self.last_request_times) >= self.rate_limit:
                # Calculate when the oldest request will be 1 minute old
                oldest_request_time = self.last_request_times[0]
                sleep_time = 60 - (current_time - oldest_request_time)

                if sleep_time > 0:
                    self.logger.debug(
                        f"Rate limit reached, sleeping for {sleep_time:.1f}s"
                    )
                    # Release lock during sleep to allow other tasks to check rate limit
                    pass  # We'll sleep outside the lock
            else:
                sleep_time = 0

            self.last_request_times.append(current_time)

        # Sleep outside the lock to not block other rate limit checks
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    async def save_results(
        self, results_generator: AsyncGenerator[Dict, None], source_url: str = None
    ):
        """Collect results from generator and save to JSON file."""
        results = {
            "analysis_metadata": {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "gemini_model_used": self.gemini_model,
                "source_data_url": source_url,
                "total_properties_processed": 0,
            },
            "properties": [],
            "processing_summary": {
                "successful_analyses": 0,
                "failed_analyses": 0,
                "total_images_processed": 0,
                "rate_limit_errors": 0,
            },
        }

        # Collect all results from generator
        async for result in results_generator:
            results["properties"].append(result)

            # Update counters
            if result["processing_info"]["status"] == "success":
                results["processing_summary"]["successful_analyses"] += 1
            else:
                results["processing_summary"]["failed_analyses"] += 1
                if "429" in str(result["processing_info"]["error_message"]):
                    results["processing_summary"]["rate_limit_errors"] += 1

            results["processing_summary"]["total_images_processed"] += result[
                "processing_info"
            ]["images_processed"]

        # Update final metadata
        results["analysis_metadata"]["total_properties_processed"] = len(
            results["properties"]
        )

        # Save to file
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Results saved to: {self.output_file}")
            print(f"\n💾 Results saved to: {self.output_file}")
            print(
                f"📊 Processed: {results['processing_summary']['successful_analyses']}/{results['analysis_metadata']['total_properties_processed']} properties"
            )

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            print(f"❌ Failed to save results: {e}")

    def load_json_from_url(self, url: str) -> Dict:
        """Load and parse JSON data from URL."""
        try:
            self.logger.info(f"Fetching data from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            data = response.json()
            self.logger.info("Successfully loaded JSON data from URL")
            return data

        except Exception as e:
            self.logger.error(f"Failed to fetch data from URL: {e}")
            return {}

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse JSON file."""
        try:
            self.logger.info(f"Loading JSON file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.logger.info("Successfully loaded JSON file")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {e}")
            return {}


# Usage example
async def main():
    processor = PropertyImageProcessor()

    # Load data from URL
    s3_url = "https://secondbrainoldnewurls.s3.us-east-1.amazonaws.com/comparison_results_20250701_020136.json"
    data = processor.load_json_from_url(s3_url)

    # Or load from file
    # data = processor.load_json_file('paste.txt')

    if not data:
        print("❌ No data found")
        return

    # Extract properties
    properties_dict = processor.extract_properties_from_json(data)

    if not properties_dict:
        print("❌ No properties with JPEG images found")
        return

    print(f"🏠 Found {len(properties_dict)} properties to process")

    # Process all properties (generator-based)
    results_generator = processor.process_all_properties(properties_dict)

    # Save results (collects from generator) - NOW PROPERLY AWAITED
    await processor.save_results(results_generator, s3_url)


if __name__ == "__main__":
    asyncio.run(main())
