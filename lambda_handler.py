import json
import asyncio
import aiohttp
import hashlib
import boto3
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import graphblas as gb
from urllib.parse import urlparse
import re
import time
import xml.etree.ElementTree as ET


class RealEstateMatrixComparatorLambda:
    def __init__(self, source_bucket: str, output_bucket: str):
        """
        Initialize the Lambda comparator
        """
        self.source_bucket = source_bucket
        self.output_bucket = output_bucket
        self.base_s3_url = f"https://{source_bucket}.s3.amazonaws.com/"
        self.hash_to_url = {}  # BLAKE hash -> original URL mapping
        self.url_to_hash = {}  # original URL -> BLAKE hash mapping
        self.s3_client = boto3.client('s3')

    async def list_bucket_files_optimized(self, session: aiohttp.ClientSession) -> List[Tuple[str, datetime]]:
        """
        Fixed version: Query S3 bucket to get ALL JSON files, then sort by date to get the 2 most recent
        Returns list of (filename, datetime) tuples sorted by date (newest first)
        """
        print(f"🔍 Querying S3 bucket '{self.source_bucket}' for all JSON files...")
        
        # Method 1: Use boto3 directly (most reliable)
        try:
            print("📡 Using boto3 to list objects...")
            
            # List all objects in the bucket
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=self.source_bucket)
            
            files_with_dates = []
            
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        filename = obj['Key']
                        
                        # Only process JSON files with the expected naming pattern
                        if filename.endswith('.json') and re.match(r'\d{8}_\d{6}\.json', filename):
                            try:
                                file_date = self.extract_date_from_filename(filename)
                                files_with_dates.append((filename, file_date))
                            except ValueError:
                                continue
            
            # Sort by date (newest first)
            files_with_dates.sort(key=lambda x: x[1], reverse=True)
            
            print(f"✅ Found {len(files_with_dates)} JSON files in bucket")
            
            # Show the files found (limit display to first 5)
            for i, (filename, file_date) in enumerate(files_with_dates[:5]):
                print(f"   {i+1}. {filename} ({file_date.strftime('%Y-%m-%d %H:%M:%S')})")
            
            if len(files_with_dates) > 5:
                print(f"   ... and {len(files_with_dates) - 5} more files")
            
            return files_with_dates
            
        except Exception as boto_error:
            print(f"❌ Boto3 method failed: {boto_error}")
            
            # Method 2: Fallback to HTTP request (fetch all, then sort)
            try:
                print("📡 Falling back to HTTP method...")
                
                # S3 bucket listing URL (XML format) - NO max-keys parameter
                bucket_list_url = f"https://{self.source_bucket}.s3.amazonaws.com/"
                
                async with session.get(bucket_list_url) as response:
                    response.raise_for_status()
                    xml_content = await response.text()
                    
                    # Parse XML response
                    root = ET.fromstring(xml_content)
                    
                    # Find all files (Contents elements)
                    files_with_dates = []
                    
                    # S3 XML namespace
                    namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
                    
                    # Look for Contents elements (files)
                    for content in root.findall('.//s3:Contents', namespace):
                        key_element = content.find('s3:Key', namespace)
                        if key_element is not None:
                            filename = key_element.text
                            
                            # Only process JSON files with the expected naming pattern
                            if filename.endswith('.json') and re.match(r'\d{8}_\d{6}\.json', filename):
                                try:
                                    file_date = self.extract_date_from_filename(filename)
                                    files_with_dates.append((filename, file_date))
                                except ValueError:
                                    continue
                    
                    # Sort by date (newest first)
                    files_with_dates.sort(key=lambda x: x[1], reverse=True)
                    
                    print(f"✅ Found {len(files_with_dates)} JSON files via HTTP fallback")
                    
                    # Show the files found (limit display to first 5)
                    for i, (filename, file_date) in enumerate(files_with_dates[:5]):
                        print(f"   {i+1}. {filename} ({file_date.strftime('%Y-%m-%d %H:%M:%S')})")
                    
                    return files_with_dates
                    
            except Exception as http_error:
                print(f"❌ HTTP fallback also failed: {http_error}")
                
                # Method 3: Last resort - use filename lexicographical sorting
                # Since your files are named YYYYMMDD_HHMMSS.json, lexicographical sorting 
                # actually works correctly for chronological order!
                print("📡 Using lexicographical sorting as last resort...")
                
                try:
                    # Get basic file list
                    response = self.s3_client.list_objects_v2(Bucket=self.source_bucket)
                    
                    if 'Contents' not in response:
                        print("⚠️  No objects found in bucket")
                        return []
                    
                    # Filter and sort files
                    json_files = []
                    for obj in response['Contents']:
                        filename = obj['Key']
                        if filename.endswith('.json') and re.match(r'\d{8}_\d{6}\.json', filename):
                            json_files.append(filename)
                    
                    # Sort lexicographically (newest last due to YYYYMMDD_HHMMSS format)
                    json_files.sort(reverse=True)  # Reverse to get newest first
                    
                    # Convert to (filename, datetime) tuples
                    files_with_dates = []
                    for filename in json_files:
                        try:
                            file_date = self.extract_date_from_filename(filename)
                            files_with_dates.append((filename, file_date))
                        except ValueError:
                            continue
                    
                    print(f"✅ Found {len(files_with_dates)} JSON files via lexicographical sorting")
                    return files_with_dates
                    
                except Exception as final_error:
                    print(f"❌ All methods failed: {final_error}")
                    return []


    def get_files_for_comparison(self, trigger_filename: str, available_files: List[Tuple[str, datetime]]) -> Tuple[Optional[str], Optional[str]]:
        """
        Fixed version: Determine latest and second-latest files based on trigger and available files
        """
        if len(available_files) < 2:
            print(f"⚠️  Need at least 2 files for comparison, found {len(available_files)}")
            return None, None
        
        # Since available_files is already sorted by date (newest first),
        # we can easily get the two most recent files
        
        # Check if trigger file is in the list
        trigger_found = any(filename == trigger_filename for filename, _ in available_files)
        
        if trigger_found:
            # Use trigger file as latest, find the next most recent
            latest_file = trigger_filename
            
            # Find second latest (first file that's not the trigger file)
            second_latest_file = None
            for filename, _ in available_files:
                if filename != trigger_filename:
                    second_latest_file = filename
                    break
        else:
            print(f"⚠️  Trigger file {trigger_filename} not found in available files")
            # Use the two most recent files
            latest_file = available_files[0][0]
            second_latest_file = available_files[1][0]
        
        if not second_latest_file:
            print(f"⚠️  Could not find second file for comparison")
            return None, None
        
        latest_url = self.base_s3_url + latest_file
        second_latest_url = self.base_s3_url + second_latest_file
        
        print(f"🎯 Selected files for comparison:")
        print(f"   Latest: {latest_file}")
        print(f"   Earlier: {second_latest_file}")
        
        return latest_url, second_latest_url
    

    def blake_hash(self, url: str) -> int:
        """Generate BLAKE2b hash for URL that fits in GraphBLAS UINT64"""
        hash_bytes = hashlib.blake2b(url.encode("utf-8"), digest_size=8).digest()
        hash_int = int.from_bytes(hash_bytes, byteorder='big')
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

    async def fetch_s3_data(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
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

                print(f"✅ Download completed: {data_size_mb:.2f} MB in {download_time:.2f}s ({data_size_mb/download_time:.2f} MB/s)")
                print(f"📊 Properties found: {len(data)}")

                return data

        except Exception as e:
            print(f"❌ Error downloading from {url}: {e}")
            return {}

    def create_hash_matrix(self, data: Dict[str, Any], dataset_name: str = "") -> Tuple[gb.Matrix, List[int], Dict[int, Dict[str, Any]]]:
        """Create GraphBLAS matrix with hash values"""
        print(f"🔧 Creating hash-value matrix for {dataset_name} dataset...")
        start_time = time.time()

        properties = []
        main_url_hashes = []
        row_details = {}
        total_images = 0

        for main_url, property_data in data.items():
            if not isinstance(property_data, dict) or "images" not in property_data:
                continue

            main_url_hash = self.add_url_mapping(main_url)
            main_url_hashes.append(main_url_hash)

            images = property_data.get("images", [])
            image_hashes = []
            image_urls = []

            for img_url in images[:100]:
                if img_url and isinstance(img_url, str) and img_url.strip():
                    if not any(skip in img_url.lower() for skip in [".svg", "logo", "remax_residential"]):
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
        ncols = 101

        matrix = gb.Matrix(gb.dtypes.UINT64, nrows=nrows, ncols=ncols)

        for row_idx, prop in enumerate(properties):
            matrix[row_idx, 0] = prop["main_url_hash"]
            for col_idx, img_hash in enumerate(prop["image_hashes"][:100]):
                if col_idx < 100:
                    matrix[row_idx, col_idx + 1] = img_hash

        matrix_time = time.time() - start_time
        sparsity = ((matrix.nrows * matrix.ncols - matrix.nvals) / (matrix.nrows * matrix.ncols) * 100)

        print(f"✅ {dataset_name} hash matrix created: {nrows} rows × {ncols} cols in {matrix_time:.2f}s")
        print(f"📈 Sparsity: {sparsity:.1f}% | Images processed: {total_images}")
        print(f"🔢 Hash values stored: {matrix.nvals:,}")

        return matrix, main_url_hashes, row_details

    def compare_hash_matrices(self, latest_matrix: gb.Matrix, latest_main_hashes: List[int], 
                            latest_row_details: Dict[int, Dict[str, Any]], earlier_matrix: gb.Matrix, 
                            earlier_main_hashes: List[int], earlier_row_details: Dict[int, Dict[str, Any]]
                            ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Compare hash-value matrices using GraphBLAS operations"""
        print("🔍 Comparing hash-value matrices using GraphBLAS...")
        start_time = time.time()

        new_updated_properties = {}
        removed_properties = {}

        latest_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(latest_main_hashes)}
        earlier_hash_to_row = {hash_val: idx for idx, hash_val in enumerate(earlier_main_hashes)}

        latest_main_set = set(latest_main_hashes)
        earlier_main_set = set(earlier_main_hashes)

        new_count = updated_count = removed_count = 0

        print(f"📊 Dataset sizes: Latest={len(latest_main_set)}, Earlier={len(earlier_main_set)}")

        # Find new properties
        new_properties = latest_main_set - earlier_main_set
        print(f"🆕 Found {len(new_properties)} completely new properties")
        
        for main_hash in new_properties:
            row_idx = latest_hash_to_row[main_hash]
            row_data = latest_row_details[row_idx]
            main_url = row_data["main_url"]
            image_urls = row_data["image_urls"]
            
            if image_urls:
                new_updated_properties[main_url] = image_urls
                new_count += 1

        # Find removed properties
        removed_props = earlier_main_set - latest_main_set
        print(f"🗑️  Found {len(removed_props)} removed properties")
        
        for main_hash in removed_props:
            row_idx = earlier_hash_to_row[main_hash]
            row_data = earlier_row_details[row_idx]
            main_url = row_data["main_url"]
            image_urls = row_data["image_urls"]
            
            removed_properties[main_url] = image_urls
            removed_count += 1

        # Find updated properties
        common_properties = latest_main_set & earlier_main_set
        print(f"🔄 Analyzing {len(common_properties)} common properties...")
        
        if common_properties and earlier_matrix.nrows > 0:
            for main_hash in common_properties:
                latest_row = latest_hash_to_row[main_hash]
                earlier_row = earlier_hash_to_row[main_hash]
                
                try:
                    latest_row_vector = latest_matrix[latest_row, :].new()
                    earlier_row_vector = earlier_matrix[earlier_row, :].new()
                    
                    latest_hashes = latest_row_vector.to_dict()
                    earlier_hashes = earlier_row_vector.to_dict()
                    
                    latest_image_hashes = {k: v for k, v in latest_hashes.items() if k != 0}
                    earlier_image_hashes = {k: v for k, v in earlier_hashes.items() if k != 0}
                    
                    if latest_image_hashes != earlier_image_hashes:
                        row_data = latest_row_details[latest_row]
                        main_url = row_data["main_url"]
                        image_urls = row_data["image_urls"]
                        
                        if image_urls:
                            new_updated_properties[main_url] = image_urls
                            updated_count += 1
                
                except Exception as e:
                    print(f"⚠️  Matrix comparison failed for property {main_hash}: {e}")

        comparison_time = time.time() - start_time
        print(f"✅ Hash matrix comparison completed in {comparison_time:.2f}s")
        print(f"📊 Results: {new_count} new, {updated_count} updated, {removed_count} removed")

        return new_updated_properties, removed_properties

    def save_results_to_s3(self, result: Dict[str, Any]) -> str:
        """Save comparison results to S3 output bucket"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_key = f"comparison_results_{timestamp}.json"
        
        try:
            self.s3_client.put_object(
                Bucket=self.output_bucket,
                Key=output_key,
                Body=json.dumps(result, indent=2),
                ContentType='application/json'
            )
            
            output_url = f"https://{self.output_bucket}.s3.amazonaws.com/{output_key}"
            print(f"💾 Results saved to: {output_url}")
            return output_url
            
        except Exception as e:
            print(f"❌ Error saving to S3: {e}")
            raise

    async def process_lambda_comparison(self, trigger_filename: str) -> Dict[str, Any]:
        """Main Lambda processing method"""
        total_start_time = time.time()
        
        print("🎯 Starting Lambda Real Estate Matrix Comparison")
        print("=" * 50)
        print(f"🔔 Triggered by file: {trigger_filename}")

        async with aiohttp.ClientSession() as session:
            # Get available files (optimized to fetch only 2 most recent)
            available_files = await self.list_bucket_files_optimized(session)
            
            if len(available_files) < 2:
                error_msg = f"Insufficient files for comparison. Found {len(available_files)}, need 2"
                print(f"❌ {error_msg}")
                return {"error": error_msg}

            # Determine which files to compare
            latest_url, earlier_url = self.get_files_for_comparison(trigger_filename, available_files)
            
            if not latest_url or not earlier_url:
                error_msg = "Could not determine files for comparison"
                print(f"❌ {error_msg}")
                return {"error": error_msg}

            print()

            # Download data asynchronously
            print("🌐 Downloading data from S3 (parallel downloads)...")
            latest_task = asyncio.create_task(self.fetch_s3_data(session, latest_url))
            earlier_task = asyncio.create_task(self.fetch_s3_data(session, earlier_url))

            latest_data, earlier_data = await asyncio.gather(latest_task, earlier_task)

        if not latest_data:
            error_msg = "No latest data available"
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        print()

        # Create hash-value matrices
        print("🔨 Building hash-value matrices...")
        latest_matrix, latest_main_hashes, latest_row_details = self.create_hash_matrix(latest_data, "Latest")
        earlier_matrix, earlier_main_hashes, earlier_row_details = self.create_hash_matrix(earlier_data, "Earlier")
        print()

        # Handle special cases
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
            new_updated_properties, removed_properties = self.compare_hash_matrices(
                latest_matrix, latest_main_hashes, latest_row_details,
                earlier_matrix, earlier_main_hashes, earlier_row_details
            )

        total_time = time.time() - total_start_time

        # Prepare result
        result = {
            "comparison_timestamp": datetime.now().isoformat(),
            "trigger_file": trigger_filename,
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
            "lambda_execution": {
                "source_bucket": self.source_bucket,
                "output_bucket": self.output_bucket,
                "trigger_file": trigger_filename
            }
        }

        # Save to output bucket
        output_url = self.save_results_to_s3(result)
        result["output_location"] = output_url

        print()
        print("🎉 Lambda Real Estate Matrix Comparison completed!")
        print(f"⏱️  Total processing time: {total_time:.2f}s")
        print(f"🔗 Total URLs hashed: {len(self.hash_to_url):,}")
        print(f"📊 Properties with changes: {len(new_updated_properties)}")
        print(f"🏠 Properties removed: {len(removed_properties)}")
        print(f"💾 Results saved to: {output_url}")

        return result


def lambda_handler(event, context):
    """AWS Lambda entry point"""
    try:
        print("🚀 Lambda function started")
        print(f"📦 Event: {json.dumps(event, indent=2)}")
        
        # Extract S3 event information
        if 'Records' not in event:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No S3 event records found'})
            }
        
        # Get the first S3 record (assuming single file upload)
        s3_record = event['Records'][0]
        
        if s3_record.get('eventSource') != 'aws:s3':
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Event is not from S3'})
            }
        
        # Extract bucket and object information
        source_bucket = s3_record['s3']['bucket']['name']
        trigger_filename = s3_record['s3']['object']['key']
        
        print(f"📁 Source bucket: {source_bucket}")
        print(f"📄 Trigger file: {trigger_filename}")
        
        # Validate that it's a JSON file with expected pattern
        if not (trigger_filename.endswith('.json') and re.match(r'\d{8}_\d{6}\.json', trigger_filename)):
            print(f"⏭️  Ignoring non-matching file: {trigger_filename}")
            return {
                'statusCode': 200,
                'body': json.dumps({'message': 'File ignored (not matching pattern)'})
            }
        
        # Initialize comparator
        output_bucket = "secondbrainoldnewurls"
        comparator = RealEstateMatrixComparatorLambda(source_bucket, output_bucket)
        
        # Run comparison
        result = asyncio.run(comparator.process_lambda_comparison(trigger_filename))
        
        if "error" in result:
            return {
                'statusCode': 500,
                'body': json.dumps({'error': result['error']})
            }
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Comparison completed successfully',
                'properties_found': result['total_properties_found'],
                'properties_removed': result['total_properties_removed'],
                'processing_time': result['processing_time_seconds'],
                'output_location': result.get('output_location')
            })
        }
        
    except Exception as e:
        print(f"❌ Lambda execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f'Lambda execution failed: {str(e)}'})
        }