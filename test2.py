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

    def blake_hash(self, url: str) -> int:
        """Generate BLAKE2b hash for URL that fits in GraphBLAS UINT64"""
        # Use digest_size=8 to get exactly 8 bytes (64 bits)
        hash_bytes = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
        # Ensure it fits in signed 64-bit range to avoid overflow
        return hash_int & 0x7FFFFFFFFFFFFFFF

    def add_url_mapping(self, url: str) -> int:
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

    def create_hash_matrix(
        self, data: Dict[str, Any], dataset_name: str = ""
    ) -> Tuple[gb.Matrix, List[int], Dict[int, Dict[str, Any]]]:
        """
        Create GraphBLAS matrix with hash values (not booleans)
        Matrix structure: [[main_url_hash, img1_hash, img2_hash, ...], ...]
        Returns: (matrix, main_url_hashes, row_details)
        """
        print(f"🔧 Creating hash-value matrix for {dataset_name} dataset...")
        start_time = time.time()

        properties = []
        main_url_hashes = []
        row_details = {}  # row_idx -> {"main_hash": int, "main_url": str, "image_hashes": List[int], "image_urls": List[str]}
        total_images = 0

        for main_url, property_data in data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue

            main_url_hash = self.add_url_mapping(main_url)
            main_url_hashes.append(main_url_hash)

            images = property_data.get("images", [])
            image_hashes = []
            image_urls = []

            for img_url in images[:100]:  # Limit to 100 images
                if img_url and isinstance(img_url, str) and img_url.strip():
                    # Skip SVG logos and other non-photo images
                    if not any(
                        skip in img_url.lower()
                        for skip in [".svg", "logo", "remax_residential"]
                    ):
                        img_hash = self.add_url_mapping(img_url)
                        image_hashes.append(img_hash)
                        image_urls.append(img_url)

            total_images += len(image_hashes)
            row_idx = len(properties)
            
            row_details[row_idx] = {
                "main_hash": main_url_hash,
                "main_url": main_url,
                "image_hashes": image_hashes,
                "image_urls": image_urls
            }

            properties.append({
                "main_url_hash": main_url_hash, 
                "image_hashes": image_hashes
            })

        if not properties:
            print(f"⚠️  No valid properties found in {dataset_name}")
            empty_matrix = gb.Matrix(gb.dtypes.UINT64, nrows=0, ncols=101)
            return empty_matrix, [], {}

        nrows = len(properties)
        ncols = 101  # Fixed: 1 main URL + 100 images

        # Create matrix with UINT64 to store hash values
        matrix = gb.Matrix(gb.dtypes.UINT64, nrows=nrows, ncols=ncols)

        for row_idx, prop in enumerate(properties):
            # Column 0: Main URL hash
            matrix[row_idx, 0] = prop["main_url_hash"]

            # Columns 1-100: Image URL hashes (0 for empty slots)
            for col_idx, img_hash in enumerate(prop["image_hashes"][:100]):
                if col_idx < 100:
                    matrix[row_idx, col_idx + 1] = img_hash

        matrix_time = time.time() - start_time
        sparsity = (
            (matrix.nrows * matrix.ncols - matrix.nvals)
            / (matrix.nrows * matrix.ncols)
            * 100
        )

        print(f"✅ {dataset_name} hash matrix created: {nrows} rows × {ncols} cols in {matrix_time:.2f}s")
        print(f"📈 Sparsity: {sparsity:.1f}% | Images processed: {total_images}")
        print(f"🔢 Hash values stored: {matrix.nvals:,}")

        return matrix, main_url_hashes, row_details

    def compare_hash_matrices(
        self,
        latest_matrix: gb.Matrix,
        latest_main_hashes: List[int],
        latest_row_details: Dict[int, Dict[str, Any]],
        earlier_matrix: gb.Matrix,
        earlier_main_hashes: List[int],
        earlier_row_details: Dict[int, Dict[str, Any]]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Compare hash-value matrices using GraphBLAS operations
        Direct hash comparison in matrices
        """
        print("🔍 Comparing hash-value matrices using GraphBLAS...")
        start_time = time.time()

        new_updated_properties = {}
        removed_properties = {}

        # Create hash-to-row mappings for efficient lookups
        latest_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(latest_main_hashes)}
        earlier_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(earlier_main_hashes)}

        # Convert to sets for main property identification
        latest_main_set = set(latest_main_hashes)
        earlier_main_set = set(earlier_main_hashes)

        new_count = 0
        updated_count = 0
        removed_count = 0

        print(f"📊 Dataset sizes: Latest={len(latest_main_set)}, Earlier={len(earlier_main_set)}")

        # 1. Find completely new properties (in latest but not in earlier)
        new_properties = latest_main_set - earlier_main_set
        print(f"🆕 Found {len(new_properties)} completely new properties")
        
        for main_hash in new_properties:
            row_idx = latest_hash_to_row[main_hash]
            row_data = latest_row_details[row_idx]
            main_url = row_data["main_url"]
            image_urls = row_data["image_urls"]
            
            if image_urls:  # Only include properties with images
                new_updated_properties[main_url] = image_urls
                new_count += 1

        # 2. Find removed properties (in earlier but not in latest)
        removed_props = earlier_main_set - latest_main_set
        print(f"🗑️  Found {len(removed_props)} removed properties")
        
        for main_hash in removed_props:
            row_idx = earlier_hash_to_row[main_hash]
            row_data = earlier_row_details[row_idx]
            main_url = row_data["main_url"]
            image_urls = row_data["image_urls"]
            
            removed_properties[main_url] = image_urls
            removed_count += 1

        # 3. Find updated properties using GraphBLAS hash matrix comparison
        common_properties = latest_main_set & earlier_main_set
        print(f"🔄 Analyzing {len(common_properties)} common properties with hash matrix comparison...")
        
        if common_properties and earlier_matrix.nrows > 0:
            for main_hash in common_properties:
                latest_row = latest_hash_to_row[main_hash]
                earlier_row = earlier_hash_to_row[main_hash]
                
                try:
                    # Extract matrix rows for direct hash comparison
                    latest_row_vector = latest_matrix[latest_row, :].new()
                    earlier_row_vector = earlier_matrix[earlier_row, :].new()
                    
                    # Convert to dictionaries for comparison (excluding column 0 - main URL)
                    latest_hashes = latest_row_vector.to_dict()
                    earlier_hashes = earlier_row_vector.to_dict()
                    
                    # Remove main URL column (column 0) from comparison
                    latest_image_hashes = {k: v for k, v in latest_hashes.items() if k != 0}
                    earlier_image_hashes = {k: v for k, v in earlier_hashes.items() if k != 0}
                    
                    # Direct hash comparison - if hash values differ, content differs
                    if latest_image_hashes != earlier_image_hashes:
                        row_data = latest_row_details[latest_row]
                        main_url = row_data["main_url"]
                        
                        # Get all current images from latest dataset
                        image_urls = row_data["image_urls"]
                        
                        if image_urls:
                            new_updated_properties[main_url] = image_urls
                            updated_count += 1
                
                except Exception as e:
                    print(f"⚠️  Matrix comparison failed for property {main_hash}: {e}")
                    # Fallback to row_details comparison
                    latest_row_data = latest_row_details[latest_row]
                    earlier_row_data = earlier_row_details[earlier_row]
                    
                    latest_img_hashes = set(latest_row_data["image_hashes"])
                    earlier_img_hashes = set(earlier_row_data["image_hashes"])
                    
                    if latest_img_hashes != earlier_img_hashes:
                        main_url = latest_row_data["main_url"]
                        image_urls = latest_row_data["image_urls"]
                        if image_urls:
                            new_updated_properties[main_url] = image_urls
                            updated_count += 1

        comparison_time = time.time() - start_time
        
        print(f"✅ Hash matrix comparison completed in {comparison_time:.2f}s")
        print(f"📊 Results: {new_count} new properties, {updated_count} updated properties, {removed_count} removed properties")
        print(f"🔗 Common properties analyzed: {len(common_properties)}")

        return new_updated_properties, removed_properties

    async def process_comparison(self, url1: str, url2: str) -> Dict[str, Any]:
        """
        Main method to process comparison using hash-value matrices
        Returns final JSON in the same format as original
        """
        total_start_time = time.time()

        print("🎯 Starting Hash-Value Matrix Comparison")
        print("=" * 45)

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

        # Create hash-value matrices (101 columns each)
        print("🔨 Building hash-value matrices (101 cols each)...")
        latest_matrix, latest_main_hashes, latest_row_details = (
            self.create_hash_matrix(latest_data, "Latest")
        )
        earlier_matrix, earlier_main_hashes, earlier_row_details = (
            self.create_hash_matrix(earlier_data, "Earlier")
        )
        print()

        # Special case handling
        if earlier_matrix.nrows == 0:
            print("⚠️  Earlier dataset is empty - treating all latest properties as new")
            new_updated_properties = {}
            for row_idx, row_data in latest_row_details.items():
                main_url = row_data["main_url"]
                image_urls = row_data["image_urls"]
                if image_urls:
                    new_updated_properties[main_url] = image_urls
            removed_properties = {}
        else:
            # Use hash matrix operations for comparison
            new_updated_properties, removed_properties = self.compare_hash_matrices(
                latest_matrix,
                latest_main_hashes,
                latest_row_details,
                earlier_matrix,
                earlier_main_hashes,
                earlier_row_details
            )

        total_time = time.time() - total_start_time

        print()
        print("🎉 Hash-Value Matrix Comparison completed!")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"🔗 Total URLs hashed: {len(self.hash_to_url):,}")
        print(f"📊 Properties with changes: {len(new_updated_properties)}")
        print(f"🏠 Properties removed: {len(removed_properties)}")
        print(f"🧮 Matrix dimensions: {latest_matrix.nrows}×{latest_matrix.ncols} (latest), {earlier_matrix.nrows}×{earlier_matrix.ncols} (earlier)")

        # Return in original JSON format
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
                "sample_mappings": {str(k): v for k, v in list(self.hash_to_url.items())[:5]},
            },
        }

        return result


async def main():
    """Example usage of the Hash-Value Matrix RealEstateMatrixComparator"""

    print("🏠 Hash-Value Matrix Real Estate Comparator")
    print("=" * 45)

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
        f"hash_matrix_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    print(f"URLs processed: {result.get('hash_mappings', {}).get('total_urls_hashed', 0):,}")

    if result.get("properties"):
        sample_prop = next(iter(result["properties"].items()))
        print(f"Sample property: {len(sample_prop[1])} images")

    if result.get("removed_properties"):
        sample_removed = next(iter(result["removed_properties"].items()))
        print(f"Sample removed property: {len(sample_removed[1])} images")

    return result


if __name__ == "__main__":
    asyncio.run(main())