import trio
import json
import logging
import os
import time
import uuid
import re
from typing import Dict, List, AsyncGenerator, Tuple

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
        rate_limit: int = 200,
        max_retries: int = 3,
        max_size_mb: float = 50.0,
        max_concurrent: int = 100,
        output_file: str = "property_analysis_results.json",
    ):
        # Load from environment variables or use provided values
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.gemini_model = gemini_model or os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
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

        # Rate limiting (200 per minute = ~3.33 per second) with thread safety
        self.last_request_times = []
        self._rate_limit_lock = trio.Lock()

        # Setup logging for MLS tracking
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

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
                    
                    # Log MLS extraction result
                    mls_info = property_details.get("mls_number", "NONE")
                    is_genuine = property_details.get("mls_is_genuine", False)
                    self.logger.info(f"Property: {property_url[:50]}... | MLS: {mls_info} ({'genuine' if is_genuine else 'generated'})")

        self.logger.info(f"Extracted {len(properties_dict)} properties with images")
        return properties_dict

    def _extract_mls_number(self, title: str) -> Tuple[str, bool]:
        """Extract MLS number from title. Returns (mls_number, is_genuine)."""
        self.logger.debug(f"MLS extraction from title: '{title}'")
        
        if not title or not isinstance(title, str):
            generated_mls = f"GEN{uuid.uuid4().hex[:8].upper()}"
            self.logger.debug(f"Empty/invalid title, generated: {generated_mls}")
            return generated_mls, False

        # Split by pipe and get the last part (most common MLS pattern)
        parts = title.split("|")
        self.logger.debug(f"Title split into {len(parts)} parts: {parts}")
        
        if len(parts) >= 2:
            potential_mls = parts[-1].strip()
        else:
            # Try other common separators
            for separator in ["-", ":", "#", "•"]:
                if separator in title:
                    parts = title.split(separator)
                    potential_mls = parts[-1].strip()
                    break
            else:
                # No separator found, check if entire title could be MLS
                potential_mls = title.strip()

        self.logger.debug(f"Potential MLS after extraction: '{potential_mls}'")

        # Clean up potential MLS - remove common prefixes/suffixes
        potential_mls = re.sub(r'^(MLS|#|ID|REF)[:.\s]*', '', potential_mls, flags=re.IGNORECASE)
        potential_mls = re.sub(r'[^\w]', '', potential_mls)  # Remove all non-alphanumeric
        
        self.logger.debug(f"Cleaned potential MLS: '{potential_mls}'")

        # Validate if it looks like a genuine MLS number
        if self._is_valid_mls(potential_mls):
            self.logger.info(f"✅ Found genuine MLS: {potential_mls}")
            return potential_mls, True
        else:
            generated_mls = f"GEN{uuid.uuid4().hex[:8].upper()}"
            self.logger.info(f"❌ No valid MLS found, generated: {generated_mls}")
            return generated_mls, False

    def _is_valid_mls(self, mls_candidate: str) -> bool:
        """Validate if a string looks like a genuine MLS number."""
        if not mls_candidate or len(mls_candidate) < 4:
            return False
        
        # Check basic alphanumeric requirement
        if not re.match(r'^[A-Za-z0-9]+$', mls_candidate):
            return False
        
        # Must have both letters and numbers (typical MLS pattern)
        has_letters = bool(re.search(r'[A-Za-z]', mls_candidate))
        has_numbers = bool(re.search(r'[0-9]', mls_candidate))
        
        # Length should be reasonable (typically 4-15 characters)
        reasonable_length = 4 <= len(mls_candidate) <= 15
        
        # Additional patterns that might indicate a genuine MLS
        common_patterns = [
            r'^[A-Z]{1,3}\d{4,}',  # Letters followed by numbers (e.g., AB123456)
            r'^\d{4,}[A-Z]{1,3}',  # Numbers followed by letters (e.g., 123456AB)
            r'^[A-Z0-9]{6,}$',     # Mixed alphanumeric, reasonable length
        ]
        
        pattern_match = any(re.match(pattern, mls_candidate.upper()) for pattern in common_patterns)
        
        is_valid = has_letters and has_numbers and reasonable_length and pattern_match
        
        self.logger.debug(f"MLS validation for '{mls_candidate}': letters={has_letters}, numbers={has_numbers}, length={reasonable_length}, pattern={pattern_match} -> {is_valid}")
        
        return is_valid

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

        # Extract MLS number from title - IMPROVED LOGIC
        title = property_data.get("title", "")
        mls_number, is_genuine = self._extract_mls_number(title)
        details["mls_number"] = mls_number
        details["mls_is_genuine"] = is_genuine

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

        # IMPROVED: Always include MLS number with context about whether it's genuine
        mls_number = property_details.get("mls_number", "Unknown")
        is_genuine = property_details.get("mls_is_genuine", False)
        if is_genuine:
            structured_details.append(f"MLS: {mls_number}")
        else:
            structured_details.append(f"Property ID: {mls_number} (system-generated)")

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
                # Download image with retries
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.get(image_url, timeout=30.0)
                        response.raise_for_status()
                        break
                    except Exception as e:
                        if attempt < self.max_retries:
                            wait_time = 2**attempt
                            await trio.sleep(wait_time)
                        else:
                            raise e

                # Encode to base64
                image_base64 = pybase64.b64encode(response.content).decode("utf-8")
                image_size_bytes = len(image_base64.encode("utf-8"))

                yield image_base64, image_size_bytes

            except Exception as e:
                self.logger.error(f"Failed to process image {image_url}: {e}")
                continue

    async def process_single_property(
        self,
        client: httpx.AsyncClient,
        property_url: str,
        property_data: Dict,
    ) -> Dict:
        """Process a single property with memory-efficient image handling."""
        try:
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
                    break

                images_batch.append(encoded_image)
                batch_size_bytes += size_bytes
                images_processed_count += 1

            if not images_batch:
                error_msg = "No images could be processed"
                return self._create_error_result(property_url, property_data, error_msg)

            # Auto-trigger API call when batch ready
            description = await self._call_gemini_api(
                images_batch, property_data["details"]
            )

            result = {
                "property_url": property_url,
                "property_details": self._format_property_details(
                    property_data["details"]
                ),
                "processing_info": {
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
            return self._create_error_result(property_url, property_data, error_msg)

    def _create_error_result(
        self, property_url: str, property_data: Dict, error_msg: str
    ) -> Dict:
        """Create error result structure."""
        return {
            "property_url": property_url,
            "property_details": self._format_property_details(property_data["details"]),
            "processing_info": {
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
            "mls_number": details.get("mls_number"),
            "mls_is_genuine": details.get("mls_is_genuine", False),
        }

    async def process_all_properties(
        self, properties_dict: Dict[str, Dict]
    ) -> AsyncGenerator[Dict, None]:
        """Generator that processes all properties with true concurrent processing using trio nursery."""
        # Use trio's memory channel for collecting results
        send_channel, receive_channel = trio.open_memory_channel(0)

        async def process_with_semaphore(
            client, property_url, property_data, semaphore
        ):
            async with semaphore:
                result = await self.process_single_property(
                    client, property_url, property_data
                )
                # Use send_nowait to avoid blocking and handle closed channel gracefully
                try:
                    send_channel.send_nowait(result)
                except trio.ClosedResourceError:
                    # Channel is closed, just ignore - this can happen during shutdown
                    pass

        async def run_all_tasks():
            """Run all property processing tasks in a nursery."""
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with trio.open_nursery() as nursery:
                    # Create semaphore for concurrency control
                    semaphore = trio.Semaphore(self.max_concurrent)

                    # Start all property processing tasks
                    for property_url, property_data in properties_dict.items():
                        nursery.start_soon(
                            process_with_semaphore,
                            client,
                            property_url,
                            property_data,
                            semaphore,
                        )
            # When nursery exits, all tasks are complete - close the send channel
            await send_channel.aclose()

        try:
            # Start the task runner in the background
            async with trio.open_nursery() as main_nursery:
                main_nursery.start_soon(run_all_tasks)

                # Yield results as they arrive
                async with receive_channel:
                    async for result in receive_channel:
                        yield result

        finally:
            pass

    async def _call_gemini_api(
        self, images_base64: List[str], property_details: Dict
    ) -> str:
        """Call Gemini API with retry logic, 300s timeout, and rate limiting - fully async and non-blocking."""
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

                # Use trio's timeout with thread pool for blocking API calls
                with trio.move_on_after(300) as cancel_scope:
                    response = await trio.to_thread.run_sync(
                        self._sync_gemini_call,
                        contents,
                    )

                # Check if the operation was cancelled due to timeout
                if cancel_scope.cancelled_caught:
                    raise TimeoutError("Gemini API call timed out after 300 seconds")

                return response.text

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    await trio.sleep(wait_time)
                else:
                    self.logger.error(f"All API attempts failed: {e}")
                    raise e

    def _sync_gemini_call(self, contents):
        """Synchronous Gemini API call to be run in trio thread pool."""
        return self.client.models.generate_content(
            model=self.gemini_model,
            contents=contents,
        )

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
                    pass  # We'll sleep outside the lock
            else:
                sleep_time = 0

            self.last_request_times.append(current_time)

        # Sleep outside the lock to not block other rate limit checks
        if sleep_time > 0:
            await trio.sleep(sleep_time)

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
                "genuine_mls_count": 0,
                "generated_mls_count": 0,
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

            # Track MLS statistics
            if result["property_details"].get("mls_is_genuine", False):
                results["processing_summary"]["genuine_mls_count"] += 1
            else:
                results["processing_summary"]["generated_mls_count"] += 1

        # Update final metadata
        results["analysis_metadata"]["total_properties_processed"] = len(
            results["properties"]
        )

        # Log final MLS statistics
        genuine_count = results["processing_summary"]["genuine_mls_count"]
        generated_count = results["processing_summary"]["generated_mls_count"]
        total_count = results["analysis_metadata"]["total_properties_processed"]
        
        self.logger.info(f"🏷️ MLS SUMMARY: {genuine_count} genuine, {generated_count} generated out of {total_count} total properties")

        # Save to file
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Results saved to: {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")

    def load_json_from_url(self, url: str) -> Dict:
        """Load and parse JSON data from URL."""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to fetch data from URL: {e}")
            return {}

    def load_json_file(self, file_path: str) -> Dict:
        """Load and parse JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load JSON file: {e}")
            return {}


# Usage example
async def main():
    processor = PropertyImageProcessor()

    # Load data from URL
    s3_url = "https://secondbrainoldnewurls.s3.us-east-1.amazonaws.com/comparison_results_20250701_020136.json"
    data = processor.load_json_from_url(s3_url)

    if not data:
        return

    # Extract properties
    properties_dict = processor.extract_properties_from_json(data)

    if not properties_dict:
        return

    # Process all properties (generator-based)
    results_generator = processor.process_all_properties(properties_dict)

    # Save results (collects from generator)
    await processor.save_results(results_generator, s3_url)


if __name__ == "__main__":
    trio.run(main)
