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

def find_house_in_yesterday(house_hash, yesterday_main_col):
    """
    Pure GraphBLAS: Find if house_hash exists in yesterday's main column
    Returns: (found, row_index)
    """
    # Create a vector with the target hash value at all positions
    target_vector = gb.Vector.full(yesterday_main_col.size, house_hash, dtype=gb.dtypes.INT64)
    
    # Element-wise comparison: where yesterday_main_col == house_hash
    match_mask = yesterday_main_col.ewise_mult(target_vector, gb.binary.eq)
    
    # Check if any match exists
    match_indices = match_mask.to_values()
    if len(match_indices[0]) > 0:
        return True, match_indices[0][0]  # Found, return first match index
    else:
        return False, None

def merge_image_vectors_graphblas(today_images, yesterday_images):
    """
    Pure GraphBLAS: Merge two image vectors (union operation)
    """
    # Create masks for non-zero elements
    today_mask = today_images.apply(gb.unary.one, gb.dtypes.BOOL)
    yesterday_mask = yesterday_images.apply(gb.unary.one, gb.dtypes.BOOL)
    
    # Union: take today's images where they exist, otherwise yesterday's
    # Use ewise_add with FIRST operator to prioritize today's values
    merged = today_images.ewise_add(yesterday_images, gb.binary.first)
    
    return merged

def vectors_are_equal(vec1, vec2):
    """
    Pure GraphBLAS: Check if two vectors are identical
    """
    # Element-wise comparison
    diff_mask = vec1.ewise_mult(vec2, gb.binary.ne)
    
    # If any differences exist, vectors are not equal
    has_difference = diff_mask.reduce(gb.monoid.lor) if diff_mask.nvals > 0 else False
    
    return not has_difference

def find_changed_houses_pure_graphblas(today_matrix, yesterday_matrix):
    """
    Pure GraphBLAS implementation - no numpy operations
    """
    print("Processing matrices with pure GraphBLAS...")
    print(f"Today matrix shape: {today_matrix.shape}")
    print(f"Yesterday matrix shape: {yesterday_matrix.shape}")
    
    # Extract main URL columns using GraphBLAS
    today_main_col = today_matrix[:, 0]
    yesterday_main_col = yesterday_matrix[:, 0]
    
    print(f"\nToday's house hashes: {[today_main_col[i].value for i in range(today_main_col.size)]}")
    print(f"Yesterday's house hashes: {[yesterday_main_col[i].value for i in range(yesterday_main_col.size)]}")
    
    # Collect changed rows using GraphBLAS matrices
    changed_row_indices = []
    changed_row_data = []
    
    # Process each house in today's matrix
    for i in range(today_matrix.nrows):
        today_house_hash = today_main_col[i].value
        if today_house_hash == 0:  # Skip empty rows
            continue
            
        print(f"\nProcessing house {today_house_hash}...")
        
        # Find if this house existed yesterday using pure GraphBLAS
        found, yesterday_row_idx = find_house_in_yesterday(today_house_hash, yesterday_main_col)
        
        if not found:
            # NEW HOUSE: Take entire row from today
            print(f"  → NEW HOUSE: {today_house_hash}")
            today_row = today_matrix[i, :]
            changed_row_indices.append(i)
            # Convert row to list for building final matrix
            row_values = [today_row[j].value for j in range(today_row.size)]
            changed_row_data.append(row_values)
            
        else:
            # EXISTING HOUSE: Check if images changed using GraphBLAS
            print(f"  → EXISTING HOUSE: {today_house_hash}, checking images...")
            
            # Extract image vectors (columns 1 onwards)
            today_images = today_matrix[i, 1:]
            yesterday_images = yesterday_matrix[yesterday_row_idx, 1:]
            
            today_img_list = [today_images[j].value for j in range(today_images.size)]
            yesterday_img_list = [yesterday_images[j].value for j in range(yesterday_images.size)]
            
            print(f"    Today's images: {today_img_list}")
            print(f"    Yesterday's images: {yesterday_img_list}")
            
            # Compare image vectors using GraphBLAS
            if not vectors_are_equal(today_images, yesterday_images):
                print("    → IMAGES CHANGED: merging with GraphBLAS...")
                
                # Merge images using GraphBLAS
                merged_images = merge_image_vectors_graphblas(today_images, yesterday_images)
                merged_img_list = [merged_images[j].value for j in range(merged_images.size)]
                
                print(f"    → Merged images: {merged_img_list}")
                
                # Build complete row: [main_url_hash] + [merged_images]
                complete_row = [today_house_hash] + merged_img_list
                changed_row_data.append(complete_row)
            else:
                print("    → IMAGES SAME: skipping...")
    
    # Build final result matrix using GraphBLAS
    if changed_row_data:
        print(f"\nBuilding final matrix with {len(changed_row_data)} changed rows...")
        for i, row in enumerate(changed_row_data):
            print(f"  Row {i+1}: {row}")
        
        # Convert to GraphBLAS matrix
        result_matrix = gb.Matrix.from_dense(np.array(changed_row_data, dtype=np.int64))
        return result_matrix
    else:
        print("\nNo changes found!")
        # Return empty matrix with same column count
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
        # Extract row using GraphBLAS
        row_vector = changes_matrix[i, :]
        main_url_hash = str(int(row_vector[0].value))
        image_hashes = [str(int(row_vector[j].value)) for j in range(1, row_vector.size) if row_vector[j].value != 0]
        
        main_url = main_keyval.get(main_url_hash)
        image_urls = [image_keyval.get(h) for h in image_hashes if image_keyval.get(h)]
        
        if main_url:
            final_json[main_url] = image_urls
            print(f"  {main_url}: {image_urls}")
    
    return final_json

# Run the demo
if __name__ == "__main__":
    print("=== Pure GraphBLAS Real Estate Demo ===\n")
    
    # Create demo matrices
    today_matrix, yesterday_matrix = create_demo_matrices()
    
    print("Today's Matrix:")
    print(today_matrix.to_dense())
    print("\nYesterday's Matrix:")  
    print(yesterday_matrix.to_dense())
    print("\n" + "="*50)
    
    # Find changed houses using pure GraphBLAS
    changes_matrix = find_changed_houses_pure_graphblas(today_matrix, yesterday_matrix)
    
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
