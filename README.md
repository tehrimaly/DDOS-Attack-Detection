#  AI-Powered DDoS Detection Dashboard

A machine learning-based network intrusion detection system with an interactive Streamlit dashboard. This project analyzes network flows to classify traffic as benign or malicious across 8 different DDoS attack types.

<img width="1450" height="682" alt="WhatsApp Image 2026-04-27 at 8 24 16 PM" src="https://github.com/user-attachments/assets/0076a468-617a-4df7-b730-d7535f29ef47" />
<img width="1920" height="938" alt="image" src="https://github.com/user-attachments/assets/e96ba286-bad1-4a8c-acb1-98168078ee83" />
<img width="1600" height="900" alt="image" src="https://github.com/user-attachments/assets/9c3ea4a8-739f-4ded-8e6f-53f36867c70a" />



##  Model Performance & Metrics
The core detection engine is powered by an **XGBoost Classifier** that was trained on over 125,000 network flows.

* **Overall Accuracy:** 99.57%
* **Benign Detection Rate:** 99.91%
* **Attack Detection Rate:** 66.60% (Specific to Syn attacks)
* **Training Time:** ~110 seconds

##  Technical Details
* **Algorithm:** XGBoost Classifier
* **Hyperparameters:** 300 Trees, Max Depth: 15, Learning Rate: 0.05
* **Features:** 77 Network Flow Features
* **Training Samples:** 125,170 flows
* **Testing Samples:** 306,201 flows

##  Classification Categories (9 Classes)
The model successfully classifies network traffic into the following 9 categories:
1. Benign (Normal Traffic)
2. LDAP
3. MSSQL
4. NetBIOS
5. Portmap
6. Syn
7. UDP
8. UDPLag
9. Unknown

##  Note on Datasets
*Due to GitHub's file size limits, the massive `.parquet` and `.csv` datasets used to train this model are not included in this repository. The repository contains the source code for the machine learning model, data processing, and the Streamlit interface.*
## Dataset
 CIC-DDoS2019
##  How to Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Arisha18-glitch/DDOS-DETECTION.git
