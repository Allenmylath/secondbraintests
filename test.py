import json
import asyncio
import aiohttp
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any
import graphblas as gb
from urllib.parse import urlparse
import re
import time


class RealEstateMatrixComparator:
    def __init__(self, base_s3_url: str):
        """
        Initialize the comparator with base S3 URL pattern
        Example: "https://secondbrainscrape.s3.us-east-1.amazonaws.com/"
        """
        self.base_s3_url = base_s3_url
        self.hash_to_url = {}  # BLAKE hash -> original URL mapping
        self.url_to_hash = {}  # original URL -> BLAKE hash mapping

    def blake_hash(self, url: str) -> str:
        """Generate BLAKE2b hash for URL"""
        return hashlib.blake2b(url.encode("utf-8")).hexdigest()

    def add_url_mapping(self, url: str) -> str:
        """Add URL to hash mapping and return hash"""
        url_hash = self.blake_hash(url)
        self.hash_to_url[url_hash] = url
        self.url_to_hash[url] = url_hash
        return url_hash

    def extract_date_from_filename(self, filename: str) -> datetime:
        """Extract date from filename like '20250627_020118.json'"""
        match = re.search(r"(\d{8}_\d{6})\.json", filename)
        if match:
            date_str = match.group(1)
            return datetime.strptime(date_str, "%Y%m%d_%H%M%S")
        raise ValueError(f"Cannot extract date from filename: {filename}")

    async def fetch_s3_data(
        self, session: aiohttp.ClientSession, url: str
    ) -> Dict[str, Any]:
        """Fetch JSON data from S3 URL asynchronously"""
        print(f"📥 Starting download from: {url}")
        start_time = time.time()

        try:
            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.read()
                data = json.loads(content)

                download_time = time.time() - start_time
                data_size_mb = len(content) / (1024 * 1024)

                print(
                    f"✅ Download completed: {data_size_mb:.2f} MB in {download_time:.2f}s ({data_size_mb/download_time:.2f} MB/s)"
                )
                print(f"📊 Properties found: {len(data)}")

                return data

        except Exception as e:
            print(f"❌ Error downloading from {url}: {e}")
            return {}

    def find_latest_files(self, url1: str, url2: str) -> Tuple[str, str]:
        """Determine which URL is latest and which is earlier based on filename dates"""
        try:
            filename1 = urlparse(url1).path.split("/")[-1]
            filename2 = urlparse(url2).path.split("/")[-1]

            date1 = self.extract_date_from_filename(filename1)
            date2 = self.extract_date_from_filename(filename2)

            if date1 > date2:
                return url1, url2  # latest, earlier
            else:
                return url2, url1  # latest, earlier

        except Exception:
            return url1, url2

    def create_property_matrix(
        self, data: Dict[str, Any], dataset_name: str = ""
    ) -> Tuple[gb.Matrix, List[str], Dict[int, List[str]]]:
        """
        Create GraphBLAS sparse matrix from property data
        Returns: (matrix, list of main_url_hashes for row mapping, row_to_image_hashes mapping)
        """
        print(f"🔧 Creating matrix for {dataset_name} dataset...")
        start_time = time.time()

        properties = []
        main_url_hashes = []
        row_to_image_hashes = {}
        total_images = 0

        for main_url, property_data in data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue

            main_url_hash = self.add_url_mapping(main_url)
            main_url_hashes.append(main_url_hash)

            images = property_data.get("images", [])
            image_hashes = []

            for img_url in images[:100]:  # Limit to 100
                if img_url and isinstance(img_url, str) and img_url.strip():
                    # Skip SVG logos and other non-photo images
                    if not any(
                        skip in img_url.lower()
                        for skip in [".svg", "logo", "remax_residential"]
                    ):
                        img_hash = self.add_url_mapping(img_url)
                        image_hashes.append(img_hash)

            total_images += len(image_hashes)
            row_idx = len(properties)
            row_to_image_hashes[row_idx] = image_hashes

            properties.append(
                {"main_url_hash": main_url_hash, "image_hashes": image_hashes}
            )

        if not properties:
            print(f"⚠️  No valid properties found in {dataset_name}")
            empty_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=0, ncols=101)
            return empty_matrix, [], {}

        nrows = len(properties)
        ncols = 101  # 1 for main URL + 100 for images

        matrix = gb.Matrix(gb.dtypes.BOOL, nrows=nrows, ncols=ncols)

        for row_idx, prop in enumerate(properties):
            matrix[row_idx, 0] = True  # Main URL presence

            # Columns 1-100: image URLs
            for col_idx, img_hash in enumerate(prop["image_hashes"][:100]):
                if col_idx < 100:
                    matrix[row_idx, col_idx + 1] = True

        matrix_time = time.time() - start_time
        sparsity = (
            (matrix.nrows * matrix.ncols - matrix.nvals)
            / (matrix.nrows * matrix.ncols)
            * 100
        )

        print(
            f"✅ {dataset_name} matrix created: {nrows} rows × {ncols} cols in {matrix_time:.2f}s"
        )
        print(f"📈 Sparsity: {sparsity:.1f}% | Images processed: {total_images}")

        return matrix, main_url_hashes, row_to_image_hashes

    def compare_matrices(
        self,
        latest_matrix: gb.Matrix,
        latest_main_hashes: List[str],
        latest_row_to_images: Dict[int, List[str]],
        earlier_matrix: gb.Matrix,
        earlier_main_hashes: List[str],
        earlier_row_to_images: Dict[int, List[str]],
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Compare matrices and return new/updated properties AND removed properties
        Returns: (new_updated_properties, removed_properties)
        """
        print("🔍 Comparing matrices...")
        start_time = time.time()

        new_updated_properties = {}
        removed_properties = {}
        new_properties_count = 0
        updated_properties_count = 0
        removed_properties_count = 0

        latest_hash_set = set(latest_main_hashes)
        earlier_hash_set = set(earlier_main_hashes)
        earlier_hash_to_row = {
            hash_val: idx for idx, hash_val in enumerate(earlier_main_hashes)
        }

        # Process latest properties (new/updated)
        for latest_row_idx, main_url_hash in enumerate(latest_main_hashes):
            main_url = self.hash_to_url[main_url_hash]

            if main_url_hash not in earlier_hash_set:
                # New property - add all its images
                image_hashes = latest_row_to_images.get(latest_row_idx, [])
                image_urls = [
                    self.hash_to_url[hash_val]
                    for hash_val in image_hashes
                    if hash_val in self.hash_to_url
                ]
                new_updated_properties[main_url] = image_urls
                new_properties_count += 1

            else:
                # Existing property - check for new images
                earlier_row_idx = earlier_hash_to_row[main_url_hash]

                latest_image_hashes = set(latest_row_to_images.get(latest_row_idx, []))
                earlier_image_hashes = set(
                    earlier_row_to_images.get(earlier_row_idx, [])
                )

                if latest_image_hashes:
                    if not earlier_image_hashes:
                        # No earlier images, treat all latest as new
                        all_image_urls = [
                            self.hash_to_url[hash_val]
                            for hash_val in latest_image_hashes
                            if hash_val in self.hash_to_url
                        ]
                        new_updated_properties[main_url] = all_image_urls
                        updated_properties_count += 1
                    else:
                        # Find new images
                        new_image_hashes = latest_image_hashes - earlier_image_hashes

                        if new_image_hashes:
                            # Merge all images (existing + new), limit to 100
                            all_image_hashes = list(
                                latest_image_hashes | earlier_image_hashes
                            )[:100]
                            all_image_urls = [
                                self.hash_to_url[hash_val]
                                for hash_val in all_image_hashes
                                if hash_val in self.hash_to_url
                            ]
                            new_updated_properties[main_url] = all_image_urls
                            updated_properties_count += 1

        # Process removed properties (in earlier but not in latest)
        for earlier_row_idx, main_url_hash in enumerate(earlier_main_hashes):
            if main_url_hash not in latest_hash_set:
                main_url = self.hash_to_url[main_url_hash]
                image_hashes = earlier_row_to_images.get(earlier_row_idx, [])
                image_urls = [
                    self.hash_to_url[hash_val]
                    for hash_val in image_hashes
                    if hash_val in self.hash_to_url
                ]
                removed_properties[main_url] = image_urls
                removed_properties_count += 1

        comparison_time = time.time() - start_time
        common_properties = len(latest_hash_set & earlier_hash_set)

        print(f"✅ Matrix comparison completed in {comparison_time:.2f}s")
        print(
            f"📊 Results: {new_properties_count} new properties, {updated_properties_count} updated properties, {removed_properties_count} removed properties"
        )
        print(f"🔗 Common properties: {common_properties}")

        return new_updated_properties, removed_properties

    async def process_comparison(self, url1: str, url2: str) -> Dict[str, Any]:
        """
        Main method to process comparison between two S3 URLs
        Returns final JSON with new/updated properties AND removed properties
        """
        total_start_time = time.time()

        print("🎯 Starting Real Estate Data Comparison")
        print("=" * 50)

        # Determine file order
        print("📅 Determining file chronology...")
        latest_url, earlier_url = self.find_latest_files(url1, url2)
        print(f"📄 Latest file: {latest_url.split('/')[-1]}")
        print(f"📄 Earlier file: {earlier_url.split('/')[-1]}")
        print()

        # Download data asynchronously
        print("🌐 Downloading data from S3 (parallel downloads)...")
        async with aiohttp.ClientSession() as session:
            latest_task = asyncio.create_task(self.fetch_s3_data(session, latest_url))
            earlier_task = asyncio.create_task(self.fetch_s3_data(session, earlier_url))

            latest_data, earlier_data = await asyncio.gather(latest_task, earlier_task)

        if not latest_data:
            return {"error": "No latest data available"}

        print()

        # Create matrices
        print("🔨 Building sparse matrices...")
        latest_matrix, latest_main_hashes, latest_row_to_images = (
            self.create_property_matrix(latest_data, "Latest")
        )
        earlier_matrix, earlier_main_hashes, earlier_row_to_images = (
            self.create_property_matrix(earlier_data, "Earlier")
        )
        print()

        # Special case handling
        if earlier_matrix.nrows == 0:
            print("⚠️  Earlier dataset is empty - treating all latest properties as new")
            new_updated_properties = {}
            for row_idx, main_url_hash in enumerate(latest_main_hashes):
                main_url = self.hash_to_url[main_url_hash]
                image_hashes = latest_row_to_images.get(row_idx, [])
                image_urls = [
                    self.hash_to_url[hash_val]
                    for hash_val in image_hashes
                    if hash_val in self.hash_to_url
                ]
                if image_urls:  # Only include properties that have images
                    new_updated_properties[main_url] = image_urls
            removed_properties = {}
        else:
            new_updated_properties, removed_properties = self.compare_matrices(
                latest_matrix,
                latest_main_hashes,
                latest_row_to_images,
                earlier_matrix,
                earlier_main_hashes,
                earlier_row_to_images,
            )

        total_time = time.time() - total_start_time

        print()
        print("🎉 Comparison completed!")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"🔗 Total URLs hashed: {len(self.hash_to_url):,}")
        print(f"📊 Properties with changes: {len(new_updated_properties)}")
        print(f"🏠 Properties removed: {len(removed_properties)}")

        result = {
            "comparison_timestamp": datetime.now().isoformat(),
            "latest_file": latest_url,
            "earlier_file": earlier_url,
            "total_properties_found": len(new_updated_properties),
            "total_properties_removed": len(removed_properties),
            "processing_time_seconds": round(total_time, 2),
            "properties": new_updated_properties,
            "removed_properties": removed_properties,
            "hash_mappings": {
                "total_urls_hashed": len(self.hash_to_url),
                "sample_mappings": dict(list(self.hash_to_url.items())[:5]),
            },
        }

        return result


async def main():
    """Example usage of the RealEstateMatrixComparator"""

    print("🏠 Real Estate Data Comparator")
    print("=" * 40)

    base_s3_url = "https://secondbrainscrape.s3.us-east-1.amazonaws.com/"

    # Example URLs (replace with actual URLs)
    url1 = "https://secondbrainscrape.s3.us-east-1.amazonaws.com/20250627_020118.json"
    url2 = "https://secondbrainscrape.s3.us-east-1.amazonaws.com/20250626_020103.json"

    comparator = RealEstateMatrixComparator(base_s3_url)
    result = await comparator.process_comparison(url1, url2)

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return result

    output_filename = (
        f"property_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    print(f"💾 Saving results to: {output_filename}")
    with open(output_filename, "w") as f:
        json.dump(result, f, indent=2)

    print()
    print("📈 SUMMARY")
    print("-" * 20)
    print(f"Properties found: {result.get('total_properties_found', 0)}")
    print(f"Properties removed: {result.get('total_properties_removed', 0)}")
    print(f"Processing time: {result.get('processing_time_seconds', 0)}s")
    print(
        f"URLs processed: {result.get('hash_mappings', {}).get('total_urls_hashed', 0):,}"
    )

    if result.get("properties"):
        sample_prop = next(iter(result["properties"].items()))
        print(f"Sample property: {len(sample_prop[1])} images")

    if result.get("removed_properties"):
        sample_removed = next(iter(result["removed_properties"].items()))
        print(f"Sample removed property: {len(sample_removed[1])} images")

    return result


if __name__ == "__main__":
    asyncio.run(main())
