# 🌐 Common Linear Family: 02b - Advanced Universal Data Sanitizer

> **Dumb Student Summary (Dimag mein fit karne wali baat):**  
> Real-world data mein numbers seedhe-saadhe nahi aate, unke sath tarah-tarah ki gandagi aati hai:  
> - Kabhi dollar sign (`$`), kabhi comma (`,`), kabhi Lakh/Crore (`₹50 Lakhs`, `3.5 Cr`).  
> - Kabhi range (`$135k - $950k`) jiska beech ka number (Midpoint) nikaalna padta hai.  
> - Kabhi million multiplier (`$1.1M`) jo bina akal ke `$1.0` ban jata hai!  
> - Kabhi broker ka phone number (`Call 9922 5599`) jo galti se price ban sakta hai!  
> Yeh guide ek **Bullet-Proof Universal Sanitizer Engine** provide karta hai jo in sabhi bimariyon ko 1 second mein clean kar deta hai!

---

## 🧭 The 6 Pillars of Advanced Sanitization

```
                          ENTERPRISE SANITIZER ENGINE
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         ▼                             ▼                             ▼
1. MULTI-CURRENCY & MULTIPLIERS  2. RANGE MIDPOINT RESOLVER     3. UNIT STANDARDIZATION
- $, ₹, €, £, AUD, USD           - $135k - $950k -> $542.5k     - sqft -> sqm (x 0.0929)
- k (10^3), M (10^6), B (10^9)   - 10 to 20 sqm -> 15 sqm       - acres -> sqm (x 4046.8)
- Lakh (10^5), Crore (10^7)      - Mixed: $950k - $1.2M         - hectares -> sqm (x 10000)
         │                             │                             │
         └─────────────────────────────┼─────────────────────────────┘
                                       ▼
                       4. INTELLIGENT JUNK REJECTION
                       - Ignore phone numbers (e.g. 9922 5599)
                       - Ignore lease rates ($/sqm, $/month)
                       - Coerce "Contact Agent", "Auction", "POA" -> NaN
```

---

## 📋 Master Copy-Paste Code: Enterprise `AdvancedDataSanitizer` Class
*(Isko copy karke kisi bhi notebook ya utility file mein paste kar lo — Real Estate, Cars, Salary, E-Commerce sab jagah kaam karega!)*

```python
# ==============================================================================
# 🌐 STEP 2b: ENTERPRISE ADVANCED DATA SANITIZER (BULLET-PROOF CLEANER)
# ==============================================================================
import re
import numpy as np
import pandas as pd

class AdvancedDataSanitizer:
    """
    Enterprise-Grade Bullet-Proof Data Sanitizer
    Features:
    - Multi-currency: $, ₹, Rs, INR, €, £, AUD, USD
    - Multipliers: k, M, Million, B, Billion, Lakh, Lac, Crore, Cr
    - Range midpoints: Handles '-', '–', '—', 'to', 'between'
    - Mixed multiplier ranges: e.g. "$950k - $1.2M" -> $1,075,000
    - Unit standardizer: sqft, acres, hectares converted to standard sqm
    - Intelligent guards: Filters out phone numbers (8-10 digits) & lease rates ($/sqm)
    - Safe NaN coercion: Rejects 'Contact Agent', 'Auction', 'POA', 'Tender'
    """
    MULTIPLIERS = {
        "k": 1_000, "thousand": 1_000,
        "m": 1_000_000, "mil": 1_000_000, "mill": 1_000_000, "million": 1_000_000,
        "b": 1_000_000_000, "billion": 1_000_000_000,
        "l": 100_000, "lac": 100_000, "lacs": 100_000, "lakh": 100_000, "lakhs": 100_000,
        "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000
    }
    
    AREA_CONVERSIONS = {
        "sqm": 1.0, "m2": 1.0, "m²": 1.0, "square meter": 1.0, "square meters": 1.0,
        "sqft": 0.092903, "sq.ft": 0.092903, "sq ft": 0.092903, "ft2": 0.092903, "ft²": 0.092903,
        "acre": 4046.86, "acres": 4046.86,
        "ha": 10000.0, "hectare": 10000.0, "hectares": 10000.0
    }

    @classmethod
    def clean_price(cls, val, min_bound=10_000, max_bound=500_000_000):
        """Extracts normalized numerical price in dollars/INR with range midpoint resolution."""
        if not isinstance(val, str) or pd.isna(val):
            return np.nan
        val = val.strip()
        
        # 1. Pure junk check without numbers
        junk_patterns = r"(contact agent|call agent|auction|expression of interest|\beoi\b|tender|poa|price on application|lease only)"
        if re.search(junk_patterns, val, re.IGNORECASE) and not re.search(r"[\$₹€£\d]", val):
            return np.nan
            
        # 2. Check if purely lease $/sqm rate without total sale price
        if re.search(r"/\s*(?:sqm|m2|m²|month|week|pa|annum)", val, re.IGNORECASE) and not re.search(r"(?:sale|for sale|\$?\d{6,})", val, re.IGNORECASE):
            return np.nan

        # 3. Extract all monetary tokens with potential multipliers (e.g. $1.1M, $135,000, 950k, 3.5 Crore)
        token_pattern = r"(?:[\$₹€£]|rs\.?|inr|aud|usd)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*([a-zA-Z]+)?"
        matches = re.finditer(token_pattern, val, re.IGNORECASE)
        
        amounts = []
        for m in matches:
            num_str = m.group(1).replace(",", "")
            raw_mult = (m.group(2) or "").lower().strip()
            
            try:
                num = float(num_str)
            except ValueError:
                continue
                
            # Filter phone numbers (e.g. 9922 5599 or 0412345678)
            if len(num_str) >= 8 and raw_mult == "" and num_str.startswith(("0", "9", "8", "7", "1300", "1800")):
                continue
                
            mult = cls.MULTIPLIERS.get(raw_mult, 1.0)
            final_amt = num * mult
            
            # Guardrails check
            if min_bound <= final_amt <= max_bound:
                amounts.append(final_amt)
                
        if not amounts:
            return np.nan
        elif len(amounts) == 1:
            return amounts[0]
        else:
            # Range detected: Take Midpoint Average!
            return (min(amounts) + max(amounts)) / 2.0

    @classmethod
    def clean_area(cls, val, target_unit="sqm"):
        """Extracts normalized numerical area in square meters (sqm) with unit conversion."""
        if not isinstance(val, str) or pd.isna(val):
            return np.nan
        val = val.strip().lower()
        
        # Detect unit in string
        unit_multiplier = 1.0
        for unit_key, conv in cls.AREA_CONVERSIONS.items():
            if unit_key in val:
                unit_multiplier = conv
                break
                
        # Extract all numeric values (integers or floating decimals)
        nums = [float(x.replace(",", "")) for x in re.findall(r"(\d+(?:,\d+)*(?:\.\d+)?)", val)]
        if not nums:
            return np.nan
            
        nums = [n * unit_multiplier for n in nums]
        if len(nums) == 1:
            return nums[0]
        else:
            # Range detected: Take Midpoint Average!
            return (min(nums) + max(nums)) / 2.0

    @classmethod
    def sanitize_dataframe(cls, df, price_col="price", area_col="area"):
        """Applies advanced sanitization across the DataFrame and prints full health audit."""
        print("=" * 70)
        print("🧼 ENTERPRISE ADVANCED DATA SANITIZATION AUDIT")
        print("=" * 70)
        df_out = df.copy()
        
        if price_col in df_out.columns:
            df_out[f"{price_col}_clean"] = df_out[price_col].apply(cls.clean_price)
            valid_p = df_out[f"{price_col}_clean"].notnull().sum()
            print(f"💰 Price Sanitized  : {valid_p:,}/{len(df):,} valid sale prices extracted")
            
        if area_col in df_out.columns:
            df_out[f"{area_col}_clean"] = df_out[area_col].apply(cls.clean_area)
            valid_a = df_out[f"{area_col}_clean"].notnull().sum()
            print(f"📐 Area Sanitized   : {valid_a:,}/{len(df):,} valid floor areas normalized (in sqm)")
            
        # Filter rows with both valid price and area
        valid_mask = df_out[f"{price_col}_clean"].notnull() & df_out[f"{area_col}_clean"].notnull()
        df_clean = df_out[valid_mask].reset_index(drop=True)
        
        print("-" * 70)
        print(f"✅ READY FOR ML MODEL: {len(df_clean):,} properties with complete Price & Area!")
        print("=" * 70)
        return df_clean
```

