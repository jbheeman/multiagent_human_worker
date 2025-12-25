# Amazon Reviews Dataset - Structured Format

This directory contains processed Amazon product review data structured for recommendation and review generation tasks.

## Overview

The dataset is derived from the [Amazon Reviews 2023 dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) and has been processed to create a structured format suitable for training recommendation systems and review generation models.

## File Structure

The dataset consists of three JSONL (JSON Lines) files:

- **`training.jsonl`** - Training set (26 users per category)
- **`validation.jsonl`** - Validation set (2 users per category)  
- **`test.jsonl`** - Test set (2 users per category)

Each file contains one JSON object per line, where each object represents a single user's review history.

## Data Collection Process

### Selection Criteria

1. **5-Core Filtering**: Only users from the 5-core dataset (users with at least 5 reviews in a category)
2. **User Sampling**: 30 unique users selected per category
3. **Review Collection**: Up to 5 reviews collected per user (minimum 5 required)
4. **Category Coverage**: Multiple product categories from Amazon

### Processing Steps

1. **Step A**: Stream the 5-core CSV and collect 30 unique user IDs per category
2. **Step B**: Stream review JSONL files and collect up to 5 reviews per target user
3. **Step C**: Stream metadata JSONL files and collect product information only for ASINs that appear in the reviews

This streaming approach minimizes memory usage and allows processing of very large datasets.

## Data Format

Each line in the JSONL files is a JSON object with the following structure:

```json
{
  "user_id": "AHWCZ47A7FIYEJ7KEJG3BO3F5YXQ",
  "category": "All_Beauty",
  "history": [
    {
      "parent_asin": "B081D87QCJ",
      "rating": 4.0,
      "timestamp_ms": 1586445693499,
      "review_excerpt": "I really like these hemp and rose under eye mask patches...",
      "review_full": "These are terrific! I really like these hemp and rose...",
      "review_title": "Make your eyes feel delightful!",
      "product": {
        "title": "AZURE Hyaluronic & Retinol Anti Aging Under Eye Pads...",
        "brand": null,
        "price": 7.98,
        "main_category": "All Beauty"
      }
    },
    // ... more history reviews (typically 4)
  ],
  "heldout": {
    "parent_asin": "B07JGD2T2J",
    "rating": 4.0,
    "timestamp_ms": 1601232869958,
    "review_excerpt": "The price isn't cheap so he wanted to make sure...",
    "review_full": "My husband used this kit and took, when asked to do...",
    "review_title": "Nice grooming set",
    "product": {
      "title": "RUGGED & DAPPER Active Regimen Grooming and Skincare Set...",
      "brand": null,
      "price": null,
      "main_category": "All Beauty"
    }
  }
}
```

### Field Descriptions

#### Top-Level Fields

- **`user_id`** (string): Unique identifier for the user
- **`category`** (string): Product category (e.g., "All_Beauty", "Toys_and_Games")
- **`history`** (array): List of previous reviews (typically 4 reviews, sorted by timestamp)
- **`heldout`** (object): The most recent review, held out for evaluation

#### Review Object Fields (in `history` and `heldout`)

- **`parent_asin`** (string): Amazon product identifier
- **`rating`** (float): Star rating (1.0 to 5.0)
- **`timestamp_ms`** (integer): Review timestamp in milliseconds since epoch
- **`review_excerpt`** (string): High-signal excerpt of the review (up to 900 characters, selected using signal-based sentence scoring)
- **`review_full`** (string): Complete review text
- **`review_title`** (string): Review title/subject
- **`product`** (object): Product metadata
  - **`title`** (string): Product title
  - **`brand`** (string or null): Product brand name
  - **`price`** (float or null): Product price in USD
  - **`main_category`** (string or null): Main product category

### Review Excerpt Generation

The `review_excerpt` field contains a high-signal excerpt generated using a signal-based scoring system that prioritizes:

- Justification/contrast words (because, since, however, but, etc.)
- Decision/outcome indicators (returned, refund, recommend, etc.)
- Value/price mentions
- Product aspect keywords (quality, fit, size, etc.)
- Strong sentiment words
- Numeric information

Sentences are scored and selected to maximize information density while maintaining coherence (sentences are kept in original order).

## Train/Validation/Test Split

The split is deterministic and category-based:

- **Training**: First 26 users per category (sorted by user_id)
- **Validation**: Next 2 users per category (users 27-28)
- **Test**: Last 2 users per category (users 29-30)

This ensures:
- No data leakage between splits
- Consistent evaluation across categories
- Reproducible splits

## Categories Included

The dataset includes reviews from multiple Amazon product categories. Some large categories (50GB+) may be excluded to manage processing resources. Categories are processed independently, allowing for category-specific analysis.

## Usage Examples

### Loading the Dataset

```python
import json

# Load training data
training_data = []
with open('training.jsonl', 'r') as f:
    for line in f:
        training_data.append(json.loads(line))

# Example: Access a user's review history
user = training_data[0]
print(f"User: {user['user_id']}")
print(f"Category: {user['category']}")
print(f"History reviews: {len(user['history'])}")
print(f"Heldout review rating: {user['heldout']['rating']}")
```

### Iterating Through Reviews

```python
# Iterate through all training examples
with open('training.jsonl', 'r') as f:
    for line in f:
        example = json.loads(line)
        
        # Process history reviews
        for review in example['history']:
            asin = review['parent_asin']
            rating = review['rating']
            text = review['review_full']
            # ... your processing here
        
        # Process heldout review
        heldout = example['heldout']
        # ... evaluation here
```

### Category-Specific Analysis

```python
# Filter by category
beauty_reviews = [
    json.loads(line) for line in open('training.jsonl')
    if json.loads(line)['category'] == 'All_Beauty'
]
```

## Data Statistics

- **Users per category**: 30 (26 train + 2 val + 2 test)
- **Reviews per user**: 5 (4 history + 1 heldout)
- **Total users**: 30 × number of categories
- **Format**: JSONL (one JSON object per line)

## Notes

1. **Memory Efficiency**: The dataset was processed using streaming to handle very large source files (some categories are 30GB+)

2. **Review Excerpts**: The `review_excerpt` field uses a signal-based algorithm to extract the most informative sentences while maintaining readability

3. **Product Metadata**: Only product metadata for ASINs that appear in collected reviews is included to minimize storage

4. **Missing Values**: Some fields may be `null` (e.g., `brand`, `price`) if not available in the source data

5. **Timestamp Ordering**: Reviews in `history` are sorted by timestamp (oldest first), with `heldout` being the most recent

## Processing Scripts

- **`test_splits.py`**: Main processing script that creates the dataset
- **`run_categories.sh`**: Bash script to process categories sequentially (one per process to manage memory)

## Citation

If you use this dataset, please cite the original Amazon Reviews 2023 dataset:

```
@dataset{amazon_reviews_2023,
  title={Amazon Reviews 2023},
  author={McAuley Lab},
  year={2023},
  url={https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023}
}
```


