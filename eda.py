import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def perform_eda(df):
    """
    Exploratory Data Analysis för NSL-KDD efter cleaning.
    """
    print("="*70)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("="*70)

    # 1. Grundinformation
    print(f"Dataset shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")

    # 2. Missing & Duplicates
    print(f"\nMissing Values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")

    # 3. Label-fördelning
    print("\n" + "="*50)
    print("LABEL FÖRDELNING")
    print("="*50)
    print(df['label'].value_counts().head(15))
    
    # Normal vs Attack
    df['attack_type'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    print("\nNormal vs Attack (%):")
    print(df['attack_type'].value_counts(normalize=True).round(3) * 100)

    # Plot label distribution
    plt.figure(figsize=(10, 6))
    df['label'].value_counts().head(10).plot(kind='bar')
    plt.title('Topp 10 Attack Typer')
    plt.ylabel('Antal')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('label_distribution.png')
    plt.show()

    # 4. Numerisk statistik
    print("\n" + "="*50)
    print("NUMERISK STATISTIK (första 10 kolumner)")
    print("="*50)
    print(df.describe().iloc[:, :10].round(2))   # visa bara några kolumner

    # 5. Kategoriska variabler - HANTERA ONE-HOT ENCODING
    print("\n" + "="*50)
    print("KATEGORISKA VARIABLER (efter encoding)")
    print("="*50)

    # Kolla vilka dummy-kolumner som finns
    protocol_cols = [col for col in df.columns if col.startswith('protocol_type_')]
    flag_cols = [col for col in df.columns if col.startswith('flag_')]

    if protocol_cols:
        print("Protocol Type (one-hot):")
        for col in protocol_cols:
            print(f"  {col}: {df[col].sum()}")

    if flag_cols:
        print("\nFlag (one-hot):")
        for col in flag_cols:
            print(f"  {col}: {df[col].sum()}")

    # Service (om den är LabelEncoded)
    if 'service' in df.columns:
        print("\nService (topp 10):")
        print(df['service'].value_counts().head(10))

    # 6. Correlation Heatmap (endast numeriska)
    print("\nSkapar correlation heatmap...")
    numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['label_binary'], errors='ignore')
    
    plt.figure(figsize=(12, 8))
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap='coolwarm', annot=False)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

    print("\n✅ EDA klar! Grafer sparade som PNG-filer i mappen.")
    return df