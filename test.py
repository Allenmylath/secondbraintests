import graphblas as gb
import numpy as np
import json

def create_demo_matrices():
    """Create demo input matrices for testing"""
    
    # Today's Matrix (3 houses)
    # Row 1: house_A with img1, img2, img3
    # Row 2: house_B with img4, img5  
    # Row 3: house_C with img6, img7, img8, img9
    today_data = [
        [1001, 2001, 2002, 2003, 0, 0, 0],  # house_A: 1001, images: 2001,2002,2003
        [1002, 2004, 2005, 0, 0, 0, 0],     # house_B: 1002, images: 2004,2005
        [1003, 2006, 2007, 2008, 2009, 0, 0] # house_C: 1003, images: 2006,2007,2008,2009
    ]
    
    # Yesterday's Matrix (3 houses)  
    # Row 1: house_A with img1, img2, img3 (SAME as today)
    # Row 2: house_D with img10, img11 (DIFFERENT house)
    # Row 3: house_C with img6, img12 (DIFFERENT images)
    yesterday_data = [
        [1001, 2001, 2002, 2003, 0, 0, 0],  # house_A: same images
        [1004, 2010, 2011, 0, 0, 0, 0],     # house_D: different house  
        [1003, 2006, 2012, 0, 0, 0, 0]      # house_C: different images (2012 instead of 2007,2008,2009)
    ]
    
    # Convert to GraphBLAS matrices
    today_matrix = gb.Matrix.from_dense(np.array(today_data, dtype=np.int64))
    yesterday_matrix = gb.Matrix.from_dense(np.array(yesterday_data, dtype=np.int64))
    
    return today_matrix, yesterday_matrix

def arrays_equal(arr1, arr2):
    """Check if two image arrays are identical"""
    return np.array_equal(arr1, arr2)

def merge_image_arrays(today_imgs, yesterday_imgs):
    """Merge image arrays, keeping unique images up to 100"""
    # Convert to sets to remove duplicates, then back to list
    today_set = set(img for img in today_imgs if img != 0)
    yesterday_set = set(img for img in yesterday_imgs if img != 0)
    
    merged_set = today_set | yesterday_set  # Union
    merged_list = list(merged_set)[:100]  # Limit to 100
    
    # Pad with zeros to make same length as original
    original_length = len(today_imgs)
    while len(merged_list) < original_length:
        merged_list.append(0)
    
    return merged_list

def find_changed_houses(today_matrix, yesterday_matrix):
    """
    Returns a matrix with only new houses or houses with changed images
    """
    print("Processing matrices...")
    print(f"Today matrix shape: {today_matrix.shape}")
    print(f"Yesterday matrix shape: {yesterday_matrix.shape}")
    
    # Extract main URL columns (column 0)
    today_main_urls = today_matrix[:, 0]
    yesterday_main_urls = yesterday_matrix[:, 0]
    
    print(f"\nToday's house hashes: {today_main_urls.to_dense()}")
    print(f"Yesterday's house hashes: {yesterday_main_urls.to_dense()}")
    
    changed_rows = []
    
    # Process each house in today's matrix
    for i in range(today_matrix.nrows):
        today_house_hash = today_main_urls[i].value if today_main_urls[i].value else 0
        if today_house_hash == 0:  # Skip empty rows
            continue
            
        print(f"\nProcessing house {today_house_hash}...")
        
        # Check if this house existed yesterday
        found_yesterday_row = None
        for j in range(yesterday_matrix.nrows):
            yesterday_house_hash = yesterday_main_urls[j].value if yesterday_main_urls[j].value else 0
            if yesterday_house_hash == today_house_hash:
                found_yesterday_row = j
                break
        
        if found_yesterday_row is None:
            # NEW HOUSE: Take entire row from today
            print(f"  → NEW HOUSE: {today_house_hash}")
            today_row = today_matrix[i, :].to_dense()
            changed_rows.append(today_row)
            
        else:
            # EXISTING HOUSE: Check if images changed
            print(f"  → EXISTING HOUSE: {today_house_hash}, checking images...")
            today_images = today_matrix[i, 1:].to_dense()  # Columns 1 onwards
            yesterday_images = yesterday_matrix[found_yesterday_row, 1:].to_dense()
            
            print(f"    Today's images: {today_images}")
            print(f"    Yesterday's images: {yesterday_images}")
            
            # Compare image arrays
            if not arrays_equal(today_images, yesterday_images):
                print("    → IMAGES CHANGED: merging...")
                # Images changed: merge today + yesterday images
                merged_images = merge_image_arrays(today_images, yesterday_images)
                merged_row = [today_house_hash] + merged_images
                print(f"    → Merged images: {merged_images}")
                changed_rows.append(merged_row)
            else:
                print("    → IMAGES SAME: skipping...")
    
    # Convert to GraphBLAS matrix
    if changed_rows:
        print(f"\nFinal changed rows: {len(changed_rows)}")
        for i, row in enumerate(changed_rows):
            print(f"  Row {i+1}: {row}")
        result_matrix = gb.Matrix.from_dense(np.array(changed_rows, dtype=np.int64))
        return result_matrix
    else:
        # Return empty matrix
        print("\nNo changes found!")
        return gb.Matrix.sparse(gb.dtypes.INT64, 0, today_matrix.ncols)

def demo_keyval_conversion(changes_matrix):
    """Demo conversion using mock keyval JSON"""
    
    # Mock keyval mappings
    main_keyval = {
        "1001": "https://remax.ca/house_A",
        "1002": "https://remax.ca/house_B", 
        "1003": "https://remax.ca/house_C",
        "1004": "https://remax.ca/house_D"
    }
    
    image_keyval = {
        "2001": "img1.jpg",
        "2002": "img2.jpg", 
        "2003": "img3.jpg",
        "2004": "img4.jpg",
        "2005": "img5.jpg",
        "2006": "img6.jpg",
        "2007": "img7.jpg",
        "2008": "img8.jpg",
        "2009": "img9.jpg",
        "2010": "img10.jpg",
        "2011": "img11.jpg",
        "2012": "img12.jpg"
    }
    
    final_json = {}
    
    print("\nConverting to actual URLs...")
    for i in range(changes_matrix.nrows):
        row_data = changes_matrix[i, :].to_dense()
        main_url_hash = str(int(row_data[0]))
        image_hashes = [str(int(img)) for img in row_data[1:] if img != 0]
        
        main_url = main_keyval.get(main_url_hash)
        image_urls = [image_keyval.get(h) for h in image_hashes if image_keyval.get(h)]
        
        if main_url:
            final_json[main_url] = image_urls
            print(f"  {main_url}: {image_urls}")
    
    return final_json

# Run the demo
if __name__ == "__main__":
    print("=== GraphBLAS Real Estate Demo ===\n")
    
    # Create demo matrices
    today_matrix, yesterday_matrix = create_demo_matrices()
    
    print("Today's Matrix:")
    print(today_matrix.to_dense())
    print("\nYesterday's Matrix:")  
    print(yesterday_matrix.to_dense())
    print("\n" + "="*50)
    
    # Find changed houses
    changes_matrix = find_changed_houses(today_matrix, yesterday_matrix)
    
    print("\n" + "="*50)
    print("Final Changes Matrix:")
    if changes_matrix.nrows > 0:
        print(changes_matrix.to_dense())
    else:
        print("(empty)")
    
    # Convert to final JSON
    print("\n" + "="*50)
    final_result = demo_keyval_conversion(changes_matrix)
    
    print(f"\nFinal JSON Output:")
    print(json.dumps(final_result, indent=2))
