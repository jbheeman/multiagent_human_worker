from typing import Optional
import json
import re
import pandas as pd
from collections import defaultdict
# Cache for user dataframe to avoid reloading
_user_df_cache = None
_behavior_df_cache = None
def get_side_info(string: str) -> str:
   
    return _format_to_json(string)
    
"""
SELECT 
    session_id,
    action_type,
    click_type,
    semantic_id,
    element_meta,
    page_meta,
    url,
    products
FROM 
    filtered_action_train
WHERE 
    session_id LIKE '7f0c8207-6a6f-49cd-9d7e-17987cfafcb9_2025%'
    AND (action_type = 'click' OR products IS NOT NULL)
ORDER BY 
    timestamp
LIMIT 100;
"""
def load_behavior_info(user_id: str, parquet_path: str = "hf://datasets/NEU-HAI/OPeRA/OPeRA_filtered/action/train/train.parquet") -> Optional[str]:

    """
    Load behavior information for a specific user from the OPeRA dataset.
    """
    global _behavior_df_cache
    if _behavior_df_cache is None:
        _behavior_df_cache = pd.read_parquet(parquet_path)
    
    

    user_products = defaultdict(list)
    filtered_df = _behavior_df_cache[_behavior_df_cache['session_id'].str.startswith(f"{user_id}_2025", na=False)]
    print(len(filtered_df))
    filtered_df = filtered_df[(filtered_df['action_type'] == 'click') | (filtered_df['action_type'] == 'product_link') | (filtered_df['action_type'] == 'quantity') | (filtered_df['action_type'] == 'purchase') | (filtered_df['action_type'] == 'cart_side_bar')]
    filtered_df = filtered_df[pd.notna(filtered_df['element_meta']) & pd.notna(filtered_df['products'])] # type: ignore
    filtered_df = filtered_df.sort_values(by='timestamp')
    print(f"Filtered dataframe length: {len(filtered_df)}")


    for index, row in filtered_df.iterrows():
        first_element_meta = row['page_meta']
        json_first_element_meta = json.loads(first_element_meta)
        prod_details = json_first_element_meta.get('product_details', {})


        if isinstance(prod_details, list):
            product_dict = {}
            for item in prod_details:
                if isinstance(item, dict):
                    product_dict.update(item)
                prod_details = product_dict
        elif not isinstance(prod_details, dict):
            raise ValueError("Product details is not a dictionary")
    # print(f"Product details: {prod_details}")
        title = prod_details.get('title', '')
        price = prod_details.get('price', '')
        asin = prod_details.get('asin', '')

        product_info = {
            'title': title,
            'price': price,
            'asin': asin
        }

        key = asin if asin else title
        if key not in user_products:
            user_products[key] = product_info
        
    result = {k: v for k, v in user_products.items()}
    print(f"User products: {json.dumps(result, indent=4)}")
    return result

# if __name__ == "__main__":
# # 324934a2-5d58-49d9-bae8-545fba660731

#     user_id = "7f0c8207-6a6f-49cd-9d7e-17987cfafcb9"
#     behavior_info = load_behavior_info(user_id)
    
    
