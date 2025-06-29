import asyncio
import json
import logging
import time
from typing import Dict, List, Optional

import httpx
import pybase64
from google import genai
from google.genai import types

from prompt import PROMPT


class GeminiImageAnalyzer:
    def __init__(
        self,
        gemini_api_key: str = "AIzaSyAz-QsOto6Yrza-__IvXgJMexQ4nnu2yYI",
        rate_limit: int = 100,
        max_retries: int = 3,
        max_size_mb: float = 50.0,
        max_concurrent: int = 5
    ):
        self.gemini_api_key = gemini_api_key
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.max_size_mb = max_size_mb
        self.max_concurrent = max_concurrent
        
        # Configure Gemini client
        self.client = genai.Client(api_key=gemini_api_key)
        
        # Rate limiting
        self.last_request_times = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    async def process_properties(self, properties_dict: Dict[str, List[str]]) -> str:
        """Process properties and return JSON string with mainurl and description."""
        results = []
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_with_semaphore(client, property_url, image_urls):
            async with semaphore:
                description = await self._process_single_property(client, property_url, image_urls)
                return {"mainurl": property_url, "description": description}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create tasks but limit concurrent execution
            tasks = [
                process_with_semaphore(client, property_url, image_urls)
                for property_url, image_urls in properties_dict.items()
            ]
            
            # Process with controlled concurrency
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    if result["description"]:
                        results.append(result)
                        self.logger.info(f"Completed {result['mainurl']}")
                    else:
                        self.logger.warning(f"No description for {result['mainurl']}")
                except Exception as e:
                    self.logger.error(f"Error processing property: {e}")
        
        return json.dumps(results, indent=2)
    
    async def _process_single_property(
        self, 
        client: httpx.AsyncClient, 
        property_url: str, 
        image_urls: List[str]
    ) -> Optional[str]:
        """Process a single property's images."""
        try:
            # Download and encode images within size limit
            images_data = await self._download_and_encode_images(client, image_urls)
            
            if not images_data:
                self.logger.warning(f"No images processed for {property_url}")
                return None
            
            # Call Gemini API
            description = await self._call_gemini_api(images_data)
            self.logger.info(f"Processed {property_url}: {len(images_data)} images")
            
            return description
            
        except Exception as e:
            self.logger.error(f"Failed to process {property_url}: {e}")
            return None
    
    async def _download_and_encode_images(
        self, 
        client: httpx.AsyncClient, 
        image_urls: List[str]
    ) -> List[str]:
        """Download images and encode to base64, respecting size limit."""
        images_data = []
        total_size_bytes = 0
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        for image_url in image_urls:
            try:
                response = await client.get(image_url)
                response.raise_for_status()
                
                # Encode to base64 using pybase64 (C wrapper for speed)
                image_base64 = pybase64.b64encode(response.content).decode('utf-8')
                image_size_bytes = len(image_base64.encode('utf-8'))
                
                # Check size limit
                if total_size_bytes + image_size_bytes > max_size_bytes:
                    break
                
                images_data.append(image_base64)
                total_size_bytes += image_size_bytes
                
            except Exception as e:
                self.logger.warning(f"Failed to download {image_url}: {e}")
                continue
        
        return images_data
    
    async def _call_gemini_api(self, images_base64: List[str]) -> str:
        """Call Gemini API with retry logic and rate limiting."""
        await self._enforce_rate_limit()
        
        for attempt in range(self.max_retries + 1):
            try:
                # Prepare content
                contents = [PROMPT]
                
                # Add all images as inline data
                for img_b64 in images_base64:
                    contents.append(
                        types.Part.from_bytes(
                            data=pybase64.b64decode(img_b64),
                            mime_type='image/jpeg'
                        )
                    )
                
                # Generate content using new API
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model="gemini-2.0-flash-exp",
                    contents=contents
                )
                
                return response.text
                
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # Simple exponential backoff
                    await asyncio.sleep(wait_time)
                else:
                    raise e
    
    async def _enforce_rate_limit(self):
        """Simple rate limiting."""
        current_time = time.time()
        
        # Remove old timestamps
        self.last_request_times = [
            t for t in self.last_request_times 
            if current_time - t < 60
        ]
        
        # Wait if at limit
        if len(self.last_request_times) >= self.rate_limit:
            sleep_time = 60 - (current_time - self.last_request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.last_request_times.append(current_time)


# Example usage
async def main():
    analyzer = GeminiImageAnalyzer()  # No need to pass API key anymore
    
    properties_data = {
        "https://example.com/property1": [
            "https://example.com/image1.jpg",
            "https://example.com/image2.jpg"
        ]
    }
    
    json_results = await analyzer.process_properties(properties_data)
    print(json_results)
    
    # Output will be:
    # [
    #   {
    #     "mainurl": "https://example.com/property1",
    #     "description": "Beautiful property with..."
    #   }
    # ]


if __name__ == "__main__":
    asyncio.run(main())