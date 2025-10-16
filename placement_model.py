import pandas as pd
import numpy as np
import joblib 
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import uniform, randint 

# -------------------------- TÜRKİYE VERİSİNE DAYALI SENTETİK VERİ OLUŞTURMA FONKSİYONU --------------------------

def create_turkish_synthetic_data(N=10000):
    np.random.seed(42)
    
    # Türkiye İşgücü Piyasasını Temsilen Kategorik Özellikler
    universite_tipleri = ['Devlet (Top 5)', 'Devlet (Orta)', 'Vakıf (Yüksek)', 'Vakıf (Düşük)']
    bolumler = ['Tıp', 'Yazılım/Bilgisayar Müh.', 'İşletme/İktisat', 'Hukuk', 'Beşeri Bilimler/Sanat']
    
    # Rastgele veri üretimi
    data = {
        'Uni_Tipi': np.random.choice(universite_tipleri, N, p=[0.1, 0.45, 0.2, 0.25]),
        'Bolum': np.random.choice(bolumler, N, p=[0.05, 0.25, 0.35, 0.1, 0.25]),
        'Staj_Sayisi': np.random.randint(0, 4, N),
        'Sem1Percentage': np.round(np.random.normal(loc=70, scale=10, size=N), 2),
        'Sem2Percentage': np.round(np.random.normal(loc=72, scale=10, size=N), 2),
        'Sem3Percentage': np.round(np.random.normal(loc=73, scale=10, size=N), 2),
        'Sem4Percentage': np.round(np.random.normal(loc=75, scale=10, size=N), 2),
        'Sem1Attendance': np.random.normal(loc=85, scale=5, size=N),
        'Sem2Attendance': np.random.normal(loc=86, scale=5, size=N),
        'Sem3Attendance': np.random.normal(loc=87, scale=5, size=N),
        'Sem4Attendance': np.random.normal(loc=88, scale=5, size=N),
    }
    df = pd.DataFrame(data)
    
    # KRİTİK EKSİK ÖZELLİKLERİN SİMÜLASYONU
    df['Lang_Score'] = np.round(np.random.normal(loc=75, scale=10, size=N), 2)
    df['Cert_Count'] = np.random.randint(0, 5, N)
    
    # HEDEF OLUŞTURMA (Türkiye İstatistiklerine Bağlı Kurallar)
    def calculate_placement_prob(row):
        prob = 0.30  # Temel olasılık
        if row['Bolum'] in ['Tıp', 'Yazılım/Bilgisayar Müh.']: prob += 0.35
        elif row['Bolum'] in ['İşletme/İktisat', 'Hukuk']: prob += 0.15
        else: prob -= 0.10
            
        prob += (row['Sem4Percentage'] / 100) * 0.20 # Son dönem notu etkisi
            
        if row['Uni_Tipi'] == 'Devlet (Top 5)': prob += 0.15
        elif row['Uni_Tipi'] == 'Vakıf (Yüksek)': prob += 0.10
            
        prob += row['Staj_Sayisi'] * 0.05
        prob += row['Cert_Count'] * 0.03
        
        return max(0.0, min(1.0, prob))

    df['Olasilik'] = df.apply(calculate_placement_prob, axis=1)
    df['Internship'] = df['Olasilik'].apply(lambda x: 1 if np.random.rand() < x else 0)
    
    return df.drop(columns=['Olasilik'])

df = create_turkish_synthetic_data(N=10000)

# -------------------------- 1. ÖZELLİK MÜHENDİSLİĞİ VE ÖN İŞLEME --------------------------

percentage_cols = [col for col in df.columns if 'Percentage' in col]
attendance_cols = [col for col in df.columns if 'Attendance' in col]

df['AvgPercentage'] = df[percentage_cols].mean(axis=1)
df['AvgAttendance'] = df[attendance_cols].mean(axis=1)

df = pd.get_dummies(df, columns=['Bolum', 'Uni_Tipi'], drop_first=True)

X = df.drop('Internship', axis=1) 
y = df['Internship']           

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)       

print(f"Eğitim veri boyutu: {X_train.shape[0]}, Test veri boyutu: {X_test.shape[0]}")
print(f"Toplam özellik sayısı: {X.shape[1]}")

# -------------------------- 2. HİPERPARAMETRE OPTİMİZASYONU --------------------------

scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

lgbm = lgb.LGBMClassifier(objective='binary', 
                          metric='binary_logloss',
                          random_state=42,
                          scale_pos_weight=scale_pos_weight, 
                          n_jobs=-1)

param_distributions = {
    'n_estimators': randint(100, 1000),      
    'learning_rate': uniform(0.01, 0.2),     
    'num_leaves': randint(10, 50),
    'max_depth': [-1, 10]               
}

print("\n🔍 RANDOMIZED Search ile En İyi Hiperparametreler Aranıyor (50 deneme)...")
random_search = RandomizedSearchCV(estimator=lgbm, 
                                   param_distributions=param_distributions, 
                                   n_iter=50, 
                                   scoring='recall', 
                                   cv=3, 
                                   verbose=1, 
                                   random_state=42)
random_search.fit(X_train_scaled, y_train)

best_lgbm_model = random_search.best_estimator_

# -------------------------- 3. FINAL MODEL DEĞERLENDİRMESİ --------------------------

y_proba_tuned = best_lgbm_model.predict_proba(X_test_scaled)[:, 1]

# EŞİK DÜZELTME: Recall'ü artırmak için 0.4 kullanalım
THRESHOLD = 0.4 
y_pred_tuned = (y_proba_tuned >= THRESHOLD).astype(int) 

accuracy_tuned = accuracy_score(y_test, y_pred_tuned)

print("\n-------------------------------------------")
print(f"✅ FINAL MODEL BAŞARISI (Eşik {THRESHOLD})")
print(f"En İyi Parametreler: {random_search.best_params_}")
print("-------------------------------------------")
print(f"Model Doğruluğu (Ayarlanmış LightGBM): {accuracy_tuned:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred_tuned, target_names=['İş Bulamadı (0)', 'İş Buldu (1)']))


# -------------------------- 4. MODELİ KAYDETME --------------------------
joblib.dump(best_lgbm_model, 'lgbm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'feature_names.pkl') 

print("\n✅ Yeni Model, Ölçekleyici ve ÖZELLİK LİSTESİ başarıyla KAYDEDİLDİ.")