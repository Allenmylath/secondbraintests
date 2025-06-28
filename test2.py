import json
import asyncio
import aiohttp
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set
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
        self.global_hash_registry = set()  # Track all unique hashes across datasets

    def blake_hash(self, url: str) -> str:
        """Generate BLAKE2b hash for URL"""
        return hashlib.blake2b(url.encode("utf-8")).hexdigest()

    def add_url_mapping(self, url: str) -> str:
        """Add URL to hash mapping and return hash"""
        url_hash = self.blake_hash(url)
        self.hash_to_url[url_hash] = url
        self.url_to_hash[url] = url_hash
        self.global_hash_registry.add(url_hash)
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

    def build_global_hash_index(self, latest_data: Dict, earlier_data: Dict) -> Tuple[Dict[str, int], List[str]]:
        """
        Build a global hash index that encompasses all unique hashes from both datasets
        Returns: (hash_to_col_mapping, col_to_hash_mapping)
        """
        print("🗂️  Building global hash index...")
        
        # Collect all unique hashes from both datasets
        all_hashes = set()
        
        # Process latest data
        for main_url, property_data in latest_data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue
            
            main_url_hash = self.add_url_mapping(main_url)
            all_hashes.add(main_url_hash)
            
            images = property_data.get("images", [])
            for img_url in images[:100]:
                if img_url and isinstance(img_url, str) and img_url.strip():
                    if not any(skip in img_url.lower() for skip in [".svg", "logo", "remax_residential"]):
                        img_hash = self.add_url_mapping(img_url)
                        all_hashes.add(img_hash)
        
        # Process earlier data
        for main_url, property_data in earlier_data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue
            
            main_url_hash = self.add_url_mapping(main_url)
            all_hashes.add(main_url_hash)
            
            images = property_data.get("images", [])
            for img_url in images[:100]:
                if img_url and isinstance(img_url, str) and img_url.strip():
                    if not any(skip in img_url.lower() for skip in [".svg", "logo", "remax_residential"]):
                        img_hash = self.add_url_mapping(img_url)
                        all_hashes.add(img_hash)
        
        # Create column mappings
        sorted_hashes = sorted(list(all_hashes))
        hash_to_col = {hash_val: idx for idx, hash_val in enumerate(sorted_hashes)}
        
        print(f"📋 Global hash index: {len(sorted_hashes)} unique hashes")
        return hash_to_col, sorted_hashes

    def create_aligned_matrix(
        self, 
        data: Dict[str, Any], 
        hash_to_col: Dict[str, int], 
        ncols: int,
        dataset_name: str = ""
    ) -> Tuple[gb.Matrix, List[str], Dict[int, Dict[str, List[str]]]]:
        """
        Create GraphBLAS matrix aligned with global hash index
        Returns: (matrix, main_url_hashes, row_details)
        """
        print(f"🔧 Creating aligned matrix for {dataset_name} dataset...")
        start_time = time.time()

        properties = []
        main_url_hashes = []
        row_details = {}  # row_idx -> {"main_hash": str, "image_hashes": List[str]}
        total_images = 0

        for main_url, property_data in data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue

            main_url_hash = self.add_url_mapping(main_url)
            main_url_hashes.append(main_url_hash)

            images = property_data.get("images", [])
            image_hashes = []

            for img_url in images[:100]:
                if img_url and isinstance(img_url, str) and img_url.strip():
                    if not any(skip in img_url.lower() for skip in [".svg", "logo", "remax_residential"]):
                        img_hash = self.add_url_mapping(img_url)
                        image_hashes.append(img_hash)

            total_images += len(image_hashes)
            row_idx = len(properties)
            
            row_details[row_idx] = {
                "main_hash": main_url_hash,
                "image_hashes": image_hashes
            }

            properties.append({
                "main_url_hash": main_url_hash, 
                "image_hashes": image_hashes
            })

        if not properties:
            print(f"⚠️  No valid properties found in {dataset_name}")
            empty_matrix = gb.Matrix(gb.dtypes.BOOL, nrows=0, ncols=ncols)
            return empty_matrix, [], {}

        nrows = len(properties)
        matrix = gb.Matrix(gb.dtypes.BOOL, nrows=nrows, ncols=ncols)

        # Fill matrix using global hash index
        for row_idx, prop in enumerate(properties):
            # Set main URL column
            main_col = hash_to_col[prop["main_url_hash"]]
            matrix[row_idx, main_col] = True

            # Set image URL columns
            for img_hash in prop["image_hashes"]:
                if img_hash in hash_to_col:
                    img_col = hash_to_col[img_hash]
                    matrix[row_idx, img_col] = True

        matrix_time = time.time() - start_time
        sparsity = (
            (matrix.nrows * matrix.ncols - matrix.nvals)
            / (matrix.nrows * matrix.ncols)
            * 100
        )

        print(f"✅ {dataset_name} aligned matrix: {nrows} rows × {ncols} cols in {matrix_time:.2f}s")
        print(f"📈 Sparsity: {sparsity:.1f}% | Images processed: {total_images}")

        return matrix, main_url_hashes, row_details

    def compare_matrices_graphblas(
        self,
        latest_matrix: gb.Matrix,
        latest_main_hashes: List[str],
        latest_row_details: Dict[int, Dict[str, List[str]]],
        earlier_matrix: gb.Matrix,
        earlier_main_hashes: List[str],
        earlier_row_details: Dict[int, Dict[str, List[str]]],
        hash_to_col: Dict[str, int],
        col_to_hash: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Compare matrices using GraphBLAS operations instead of set operations
        """
        print("🔍 Comparing matrices using GraphBLAS operations...")
        start_time = time.time()

        new_updated_properties = {}
        removed_properties = {}

        # Create hash-to-row mappings for efficient lookups
        latest_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(latest_main_hashes)}
        earlier_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(earlier_main_hashes)}

        # Convert to sets for main property identification
        latest_main_set = set(latest_main_hashes)
        earlier_main_set = set(earlier_main_hashes)

        # 1. Find completely new properties (in latest but not in earlier)
        new_properties = latest_main_set - earlier_main_set
        for main_hash in new_properties:
            main_url = self.hash_to_url[main_hash]
            row_idx = latest_hash_to_row[main_hash]
            image_hashes = latest_row_details[row_idx]["image_hashes"]
            image_urls = [self.hash_to_url[h] for h in image_hashes if h in self.hash_to_url]
            if image_urls:
                new_updated_properties[main_url] = image_urls

        # 2. Find removed properties (in earlier but not in latest)
        removed_props = earlier_main_set - latest_main_set
        for main_hash in removed_props:
            main_url = self.hash_to_url[main_hash]
            row_idx = earlier_hash_to_row[main_hash]
            image_hashes = earlier_row_details[row_idx]["image_hashes"]
            image_urls = [self.hash_to_url[h] for h in image_hashes if h in self.hash_to_url]
            removed_properties[main_url] = image_urls

        # 3. Find updated properties using GraphBLAS matrix operations
        common_properties = latest_main_set & earlier_main_set
        
        if common_properties and earlier_matrix.nrows > 0:
            print(f"🔄 Analyzing {len(common_properties)} common properties with GraphBLAS...")
            
            for main_hash in common_properties:
                latest_row = latest_hash_to_row[main_hash]
                earlier_row = earlier_hash_to_row[main_hash]
                
                # Extract matrix rows as vectors for comparison
                try:
                    # Get row vectors from matrices
                    latest_row_vector = latest_matrix[latest_row, :].new()
                    earlier_row_vector = earlier_matrix[earlier_row, :].new()
                    
                    # Ensure both vectors have same size (pad if needed)
                    max_cols = max(latest_matrix.ncols, earlier_matrix.ncols)
                    
                    if latest_matrix.ncols < max_cols:
                        # Expand latest row vector
                        temp_latest = gb.Vector(gb.dtypes.BOOL, size=max_cols)
                        temp_latest[:latest_matrix.ncols] = latest_row_vector
                        latest_row_vector = temp_latest
                    
                    if earlier_matrix.ncols < max_cols:
                        # Expand earlier row vector  
                        temp_earlier = gb.Vector(gb.dtypes.BOOL, size=max_cols)
                        temp_earlier[:earlier_matrix.ncols] = earlier_row_vector
                        earlier_row_vector = temp_earlier
                    
                    # Use GraphBLAS XOR to find differences
                    diff_vector = latest_row_vector.ewise_add(earlier_row_vector, gb.binary.lxor).new()
                    
                    # Check if there are any differences
                    if diff_vector.nvals > 0:
                        # Property has changes - get all current images
                        main_url = self.hash_to_url[main_hash]
                        
                        # Get all images from latest (existing + new)
                        latest_image_hashes = latest_row_details[latest_row]["image_hashes"]
                        earlier_image_hashes = earlier_row_details[earlier_row]["image_hashes"]
                        
                        # Use set operations to find if there are actually new images
                        latest_img_set = set(latest_image_hashes)
                        earlier_img_set = set(earlier_image_hashes)
                        
                        if latest_img_set != earlier_img_set:
                            # Combine all images (prioritize latest)
                            all_image_hashes = list(latest_img_set | earlier_img_set)[:100]
                            image_urls = [self.hash_to_url[h] for h in all_image_hashes if h in self.hash_to_url]
                            if image_urls:
                                new_updated_properties[main_url] = image_urls
                
                except Exception as e:
                    print(f"⚠️  GraphBLAS operation failed for property {main_hash}: {e}")
                    # Fallback to set-based comparison for this property
                    main_url = self.hash_to_url[main_hash]
                    latest_image_hashes = set(latest_row_details[latest_row]["image_hashes"])
                    earlier_image_hashes = set(earlier_row_details[earlier_row]["image_hashes"])
                    
                    if latest_image_hashes != earlier_image_hashes:
                        all_image_hashes = list(latest_image_hashes | earlier_image_hashes)[:100]
                        image_urls = [self.hash_to_url[h] for h in all_image_hashes if h in self.hash_to_url]
                        if image_urls:
                            new_updated_properties[main_url] = image_urls

        comparison_time = time.time() - start_time
        
        print(f"✅ GraphBLAS matrix comparison completed in {comparison_time:.2f}s")
        print(f"📊 Results: {len(new_properties)} new, {len(new_updated_properties) - len(new_properties)} updated, {len(removed_properties)} removed")
        print(f"🔗 Common properties analyzed: {len(common_properties)}")

        return new_updated_properties, removed_properties

    async def process_comparison(self, url1: str, url2: str) -> Dict[str, Any]:
        """
        Main method to process comparison using GraphBLAS matrix operations
        """
        total_start_time = time.time()

        print("🎯 Starting GraphBLAS Real Estate Data Comparison")
        print("=" * 55)

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

        # Build global hash index for aligned matrices
        hash_to_col, col_to_hash = self.build_global_hash_index(latest_data, earlier_data)
        ncols = len(col_to_hash)
        print()

        # Create aligned matrices
        print("🔨 Building aligned sparse matrices...")
        latest_matrix, latest_main_hashes, latest_row_details = (
            self.create_aligned_matrix(latest_data, hash_to_col, ncols, "Latest")
        )
        earlier_matrix, earlier_main_hashes, earlier_row_details = (
            self.create_aligned_matrix(earlier_data, hash_to_col, ncols, "Earlier")
        )
        print()

        # Special case handling
        if earlier_matrix.nrows == 0:
            print("⚠️  Earlier dataset is empty - treating all latest properties as new")
            new_updated_properties = {}
            for row_idx, main_hash in enumerate(latest_main_hashes):
                main_url = self.hash_to_url[main_hash]
                image_hashes = latest_row_details[row_idx]["image_hashes"]
                image_urls = [self.hash_to_url[h] for h in image_hashes if h in self.hash_to_url]
                if image_urls:
                    new_updated_properties[main_url] = image_urls
            removed_properties = {}
        else:
            # Use GraphBLAS operations for comparison
            new_updated_properties, removed_properties = self.compare_matrices_graphblas(
                latest_matrix,
                latest_main_hashes,
                latest_row_details,
                earlier_matrix,
                earlier_main_hashes,
                earlier_row_details,
                hash_to_col,
                col_to_hash
            )

        total_time = time.time() - total_start_time

        print()
        print("🎉 GraphBLAS Comparison completed!")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"🔗 Total URLs hashed: {len(self.hash_to_url):,}")
        print(f"📊 Properties with changes: {len(new_updated_properties)}")
        print(f"🏠 Properties removed: {len(removed_properties)}")
        print(f"🧮 Matrix dimensions: {latest_matrix.nrows}×{latest_matrix.ncols} (latest), {earlier_matrix.nrows}×{earlier_matrix.ncols} (earlier)")

        result = {
            "comparison_timestamp": datetime.now().isoformat(),
            "latest_file": latest_url,
            "earlier_file": earlier_url,
            "total_properties_found": len(new_updated_properties),
            "total_properties_removed": len(removed_properties),
            "processing_time_seconds": round(total_time, 2),
            "matrix_info": {
                "latest_matrix_shape": [latest_matrix.nrows, latest_matrix.ncols],
                "earlier_matrix_shape": [earlier_matrix.nrows, earlier_matrix.ncols],
                "global_hash_columns": ncols,
                "latest_matrix_sparsity": round((latest_matrix.nrows * latest_matrix.ncols - latest_matrix.nvals) / (latest_matrix.nrows * latest_matrix.ncols) * 100, 2) if latest_matrix.nvals > 0 else 100,
                "earlier_matrix_sparsity": round((earlier_matrix.nrows * earlier_matrix.ncols - earlier_matrix.nvals) / (earlier_matrix.nrows * earlier_matrix.ncols) * 100, 2) if earlier_matrix.nvals > 0 else 100
            },
            "properties": new_updated_properties,
            "removed_properties": removed_properties,
            "hash_mappings": {
                "total_urls_hashed": len(self.hash_to_url),
                "sample_mappings": dict(list(self.hash_to_url.items())[:5]),
            },
        }

        return result


async def main():
    """Example usage of the GraphBLAS RealEstateMatrixComparator"""

    print("🏠 GraphBLAS Real Estate Data Comparator")
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
        f"graphblas_property_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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
    
    matrix_info = result.get('matrix_info', {})
    print(f"Latest matrix: {matrix_info.get('latest_matrix_shape', 'N/A')} ({matrix_info.get('latest_matrix_sparsity', 'N/A')}% sparse)")
    print(f"Earlier matrix: {matrix_info.get('earlier_matrix_shape', 'N/A')} ({matrix_info.get('earlier_matrix_sparsity', 'N/A')}% sparse)")

    if result.get("properties"):
        sample_prop = next(iter(result["properties"].items()))
        print(f"Sample property: {len(sample_prop[1])} images")

    if result.get("removed_properties"):
        sample_removed = next(iter(result["removed_properties"].items()))
        print(f"Sample removed property: {len(sample_removed[1])} images")

    return result


if __name__ == "__main__":
    asyncio.run(main())
