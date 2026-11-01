# Housing Price Prediction – Summary
**Goal:** estimate a home’s likely sale price from its attributes (beds, baths, size, type, neighborhood).
Regression problem (predicting a number).

**Data:** market listings/sales; leakage fields (e.g., price-per-sqft, listing price) removed.
**Target:** trained on log(SoldPrice) to reduce relative error; predictions converted back to dollars.

**Model:** Random Forest (tuned).
- RF_tuned_log: MAE=$569,860, R²=0.587

**Use:** input property profile → estimated sale price.
**Limits:** unusual homes/outliers and fast market shifts reduce accuracy.