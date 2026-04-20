import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def main():
    print("Loading transaction data...")
    trans_path = "workspace/transactions/transactions.csv"
    df_trans = pd.read_csv(trans_path)
    
    # Convert CSV string to list of items
    dataset = [items.split(",") for items in df_trans["Items"].tolist()]
    
    # One-hot encode the transactions for mlxtend
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    min_support = 0.05
    min_confidence = 0.5
    
    # 1. Apriori Algorithm
    print("\nRunning Apriori...")
    start_time = time.time()
    frequent_itemsets_apriori = apriori(df_encoded, min_support=min_support, use_colnames=True)
    apriori_time = time.time() - start_time
    print(f"Apriori execution time: {apriori_time:.4f} seconds")
    print(f"Apriori frequent itemsets found: {len(frequent_itemsets_apriori)}")
    
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
    print(f"Apriori rules generated: {len(rules_apriori)}")
    
    # 2. FP-Growth Algorithm
    print("\nRunning FP-Growth...")
    start_time = time.time()
    frequent_itemsets_fp = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    fp_time = time.time() - start_time
    print(f"FP-Growth execution time: {fp_time:.4f} seconds")
    print(f"FP-Growth frequent itemsets found: {len(frequent_itemsets_fp)}")
    
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence)
    print(f"FP-Growth rules generated: {len(rules_fp)}")
    
    # Evaluate and Save Results
    # Both algorithms will yield the same rules for the same thresholds. We'll use the rules from FP-Growth.
    
    # Sort rules by Lift descending
    rules_sorted = rules_fp.sort_values(by="lift", ascending=False)
    
    # Format rules for better readability
    def format_rule(row):
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        return f"{', '.join(antecedents)} -> {', '.join(consequents)}"
        
    rules_sorted['rule_string'] = rules_sorted.apply(format_rule, axis=1)
    
    # Save all rules
    rules_path = "workspace/outputs/association_rules.csv"
    rules_sorted.to_csv(rules_path, index=False)
    print(f"\nAll rules saved to {rules_path}")
    
    # Save frequent itemsets
    itemsets_path = "workspace/outputs/frequent_itemsets.csv"
    frequent_itemsets_fp['itemsets_string'] = frequent_itemsets_fp['itemsets'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets_fp.to_csv(itemsets_path, index=False)
    
    # Extract Top 5 Rules
    top_5_rules = rules_sorted.head(5)
    print("\n--- TOP 5 RULES ---")
    for idx, row in top_5_rules.iterrows():
        print(f"Rule: {row['rule_string']}")
        print(f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")
        
    # Write Comparison Report
    report_content = f"""# Performance Comparison Report

## Execution Time
- Apriori: {apriori_time:.4f} seconds
- FP-Growth: {fp_time:.4f} seconds

## Patterns Discovered (min_support={min_support}, min_confidence={min_confidence})
- Frequent Itemsets: {len(frequent_itemsets_fp)}
- Association Rules: {len(rules_fp)}

## Conclusion
FP-Growth uses an FP-Tree data structure and does not require candidate generation, making it generally faster than Apriori, especially for large datasets.
"""
    
    report_path = "workspace/outputs/comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)
    print(f"Comparison report saved to {report_path}")

if __name__ == "__main__":
    main()
