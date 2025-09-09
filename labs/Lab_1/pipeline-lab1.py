import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

print("🚀 DÉBUT DU LAB1 MLOPS - PIPELINE COMPLET")
print("⏰ Deadline: 17h00")
print("="*60)

# =================================================
# MODULE 1: COLLECTE DES DONNÉES
# =================================================

def charger_donnees():
    """Charge le dataset Breast Cancer Wisconsin"""
    from sklearn.datasets import load_breast_cancer
    donnees = load_breast_cancer(as_frame=True)
    dataset = pd.concat([donnees['data'], donnees['target']], axis=1)
    print(f"✅ Dataset chargé: {dataset.shape[0]} échantillons, {dataset.shape[1]-1} features")
    return dataset

# =================================================
# MODULE 2: ANALYSE EXPLORATOIRE DES DONNÉES
# =================================================

def analyse_exploratoire(donnees):
    """Effectue l'analyse exploratoire avec visualisations"""
    print("\n📊 ANALYSE EXPLORATOIRE EN COURS...")
    
    X = donnees.drop("target", axis=1)
    y = donnees["target"]
    
    # Configuration des graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Graphique 1: Distribution des classes
    sns.countplot(data=donnees, x='target', ax=axes[0,0])
    axes[0,0].set_title('Distribution des Classes\n(0=Malignant, 1=Bénin)')
    
    # Graphique 2: Features les plus importantes
    scores_importance = mutual_info_classif(X, y, random_state=42)
    df_importance = pd.DataFrame({
        'Feature': X.columns,
        'Score': scores_importance
    }).sort_values('Score', ascending=False).head(10)
    
    sns.barplot(data=df_importance, y='Feature', x='Score', ax=axes[0,1], palette='viridis')
    axes[0,1].set_title('Top 10 Features Importantes')
    
    # Graphique 3: Corrélation entre features principales
    features_principales = ['mean radius', 'mean texture', 'mean perimeter', 'mean area']
    matrice_correlation = donnees[features_principales].corr()
    sns.heatmap(matrice_correlation, annot=True, cmap='coolwarm', ax=axes[1,0])
    axes[1,0].set_title('Corrélation Features Principales')
    
    # Graphique 4: Distribution d'une feature par classe
    sns.boxplot(data=donnees, x='target', y='mean radius', ax=axes[1,1])
    axes[1,1].set_title('Distribution Mean Radius par Classe')
    
    plt.tight_layout()
    plt.show()
    print("✅ Analyse exploratoire terminée!")
    
    return X, y

# =================================================
# MODULE 3: INGÉNIERIE DES FEATURES
# =================================================

def ingenierie_features_basique(X):
    """Feature engineering pour modèles simples"""
    X_nouveau = X.copy()
    X_nouveau['combinaison_radius_texture'] = X['mean radius'] * X['mean texture']
    return X_nouveau

def ingenierie_features_avancee(X):
    """Feature engineering avancée pour modèles complexes"""
    X_nouveau = X.copy()
    
    # Features de base
    X_nouveau['combinaison_radius_texture'] = X['mean radius'] * X['mean texture']
    
    # Ratios pertinents
    X_nouveau['ratio_radius_texture'] = X['mean radius'] / (X['mean texture'] + 0.001)
    X_nouveau['ratio_area_perimeter'] = X['mean area'] / (X['mean perimeter'] + 0.001)
    
    # Interactions
    X_nouveau['interaction_compactness_concavity'] = X['mean compactness'] * X['mean concavity']
    
    print(f"✅ Features créées: {X_nouveau.shape[1] - X.shape[1]} nouvelles features")
    return X_nouveau

# =================================================
# MODULE 4: CONSTRUCTION DES PIPELINES
# =================================================

def creer_pipeline_logistique():
    """Pipeline pour la régression logistique"""
    return Pipeline([
        ('features', FunctionTransformer(ingenierie_features_basique)),
        ('normalisation', StandardScaler()),
        ('modele', LogisticRegression(max_iter=1000))
    ])

def creer_pipeline_avance():
    """Pipeline pour Random Forest et SVM"""
    return Pipeline([
        ('features', FunctionTransformer(ingenierie_features_avancee)),
        ('normalisation', StandardScaler()),
        ('modele', RandomForestClassifier())  # Sera remplacé par GridSearchCV
    ])

# =================================================
# MODULE 5: OPTIMISATION DES HYPERPARAMÈTRES
# =================================================

def definir_grille_parametres():
    """Définit les grilles d'hyperparamètres pour chaque modèle"""
    
    grille_logistique = [{
        'modele': [LogisticRegression(max_iter=1000)],
        'modele__C': [0.1, 1.0, 10.0]
    }]
    
    grille_avancee = [
        # Random Forest
        {
            'modele': [RandomForestClassifier(random_state=42)],
            'modele__n_estimators': [50, 100, 200],
            'modele__max_depth': [None, 10, 20]
        },
        # Support Vector Machine
        {
            'modele': [SVC()],
            'modele__C': [0.1, 1, 10],
            'modele__kernel': ['linear', 'rbf']
        }
    ]
    
    return grille_logistique, grille_avancee

# =================================================
# MODULE 6: ÉVALUATION DES MODÈLES
# =================================================

def evaluer_modele(modele, X_test, y_test, nom_modele):
    """Évalue un modèle et affiche les métriques"""
    print(f"\n📊 ÉVALUATION: {nom_modele}")
    
    predictions = modele.predict(X_test)
    precision = accuracy_score(y_test, predictions)
    
    print(f"Précision: {precision:.3f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, predictions, target_names=['Malignant', 'Bénin']))
    
    # Matrice de confusion
    matrice_conf = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrice_conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Malignant', 'Bénin'],
                yticklabels=['Malignant', 'Bénin'])
    plt.title(f'Matrice de Confusion - {nom_modele}')
    plt.ylabel('Valeurs Réelles')
    plt.xlabel('Prédictions')
    plt.show()
    
    return precision

# =================================================
# MODULE 7: FONCTION PRINCIPALE
# =================================================

def main():
    """Fonction principale qui orchestre tout le pipeline"""
    
    # 1. Chargement des données
    donnees = charger_donnees()
    
    # 2. Analyse exploratoire
    X, y = analyse_exploratoire(donnees)
    
    # 3. Division des données
    print("\n🔄 Division des données (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"✅ Train: {X_train.shape[0]} échantillons, Test: {X_test.shape[0]} échantillons")
    
    # 4. Optimisation Régression Logistique
    print("\n🎯 OPTIMISATION RÉGRESSION LOGISTIQUE...")
    pipeline_log = creer_pipeline_logistique()
    grille_log, grille_avancee = definir_grille_parametres()
    
    recherche_log = GridSearchCV(pipeline_log, grille_log, cv=5, scoring='accuracy', n_jobs=-1)
    recherche_log.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres LogReg: {recherche_log.best_params_}")
    print(f"Score CV: {recherche_log.best_score_:.3f}")
    
    # 5. Optimisation modèles avancés
    print("\n🎯 OPTIMISATION RANDOM FOREST & SVM...")
    pipeline_avance = creer_pipeline_avance()
    
    recherche_avancee = GridSearchCV(pipeline_avance, grille_avancee, cv=5, scoring='accuracy', n_jobs=-1)
    recherche_avancee.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres avancés: {recherche_avancee.best_params_}")
    print(f"Score CV: {recherche_avancee.best_score_:.3f}")
    
    # 6. Évaluation finale
    print("\n" + "="*60)
    print("📊 ÉVALUATION FINALE SUR LE JEU DE TEST")
    print("="*60)
    
    precision_log = evaluer_modele(recherche_log.best_estimator_, X_test, y_test, "Régression Logistique")
    precision_avancee = evaluer_modele(recherche_avancee.best_estimator_, X_test, y_test, "Modèle Avancé")
    
    # 7. Sélection du meilleur modèle
    if precision_log > precision_avancee:
        meilleur_modele = recherche_log.best_estimator_
        nom_meilleur = "Régression Logistique"
    else:
        meilleur_modele = recherche_avancee.best_estimator_
        nom_meilleur = "Modèle Avancé"
    
    print(f"\n🏆 MEILLEUR MODÈLE: {nom_meilleur}")
    
    # 8. Sauvegarde
    dump(meilleur_modele, 'meilleur_pipeline_lab1.joblib')
    print("💾 Modèle sauvegardé: meilleur_pipeline_lab1.joblib")
    
    print("\n🎉 LAB1 TERMINÉ AVEC SUCCÈS!")

# =================================================
# EXÉCUTION
# =================================================

if __name__ == "__main__":
    main()
