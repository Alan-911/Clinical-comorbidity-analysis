# Performance Comparison Report

## Execution Time
- Apriori: 0.0081 seconds
- FP-Growth: 0.0109 seconds

## Patterns Discovered (min_support=0.05, min_confidence=0.5)
- Frequent Itemsets: 114
- Association Rules: 144

## Conclusion
FP-Growth uses an FP-Tree data structure and does not require candidate generation, making it generally faster than Apriori, especially for large datasets.
