# 1. Install dependencies
pip install tensorflow keras numpy pandas matplotlib seaborn scikit-learn opencv-python Pillow streamlit

# 2. Download dataset from Kaggle
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
copy the dataset for windows archive/chest_xray/chest_xray

# 3. Train models
python train_model.py

# 4. Launch web app
streamlit run app.py
