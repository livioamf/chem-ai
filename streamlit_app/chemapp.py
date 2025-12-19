import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from rdkit import Chem
from rdkit.Chem import Draw

API_BASE = "http://localhost:8010"

def get_auth_headers(token):
    return {"Authorization": f"Bearer {token}"}

def predict_viscosity(token, smile_1, smile_2, fraction, temperature, architecture):
    payload = {
        "smile_1": smile_1,
        "smile_2": smile_2 if smile_2 and smile_2.strip() else None,
        "fraction": fraction if fraction is not None else None,
        "temperature": temperature,
    }
    resp = requests.post(
        f"{API_BASE}/predictions/viscosity",
        params={"architecture": architecture},
        json=payload,
        headers=get_auth_headers(token),
    )
    return resp.json()

def predict_batch(token, df, architecture):
    inputs = df.to_dict(orient="records")
    resp = requests.post(
        f"{API_BASE}/predictions/viscosity/batch",
        params={"architecture": architecture},
        json={"inputs": inputs},
        headers=get_auth_headers(token),
    )
    return resp.json()

def plot_real_pred(y_test, y_pred):
    r2 = r2_score(y_test, y_pred)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.7, label=f'R² = {r2:.3f}')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Ideal (y=x)')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title('Valores Reais vs Preditos')
    plt.legend()
    plt.grid(True)
    return plt

def show_predicao_page(token):
    """Função principal da página de predição"""
    
    st.header("🔬 Predição de Viscosidade")
    st.markdown("Faça predições de viscosidade para substâncias puras ou misturas.")
    
    # Tabs para diferentes modos de predição
    tab1, tab2 = st.tabs(["Predição única", "Predição em lote"])
    
    with tab1:
        st.subheader("Predição Individual")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smile_1 = st.text_input("SMILES 1*", placeholder="Ex: CCO (etanol)")
            smile_2 = st.text_input("SMILES 2 (opcional)", placeholder="Ex: CC(=O)O (ácido acético)")
            fraction = st.number_input(
                "Fração da molécula 2", 
                min_value=0.0, 
                max_value=1.0, 
                value=None,
                step=0.01,
                format="%.2f",
                help="Fração molar da segunda molécula (0 a 1)"
            )
        
        with col2:
            temperature = st.number_input(
                "Temperatura (K)*", 
                value=298.15,
                min_value=0.0,
                step=1.0,
                help="Temperatura em Kelvin"
            )
            architecture = st.selectbox(
                "Modelo", 
                ["base", "lora"],
                help="Arquitetura do modelo de predição"
            )
        
        # Botão de predição
        if st.button("Calcular Viscosidade", type="primary", use_container_width=True):
            if not smile_1:
                st.error("Informe o SMILES 1.")
            elif not temperature:
                st.error("Informe a temperatura.")
            else:
                with st.spinner("Calculando viscosidade..."):
                    resultado = predict_viscosity(token, smile_1, smile_2, fraction, temperature, architecture)
                    
                    if "viscosity" in resultado:
                        st.success(f"**Viscosidade estimada:** {resultado['viscosity']:.4f}")
                        
                        # Mostrar detalhes
                        with st.expander("Ver detalhes da predição"):
                            st.json(resultado)
                    else:
                        st.error(f"Erro: {resultado.get('detail', 'Erro desconhecido')}")
        
        # Visualização das moléculas
        if smile_1:
            st.divider()
            st.subheader("Visualização das Moléculas")
            
            cols_mol = st.columns(2)
            
            with cols_mol[0]:
                mol1 = Chem.MolFromSmiles(smile_1)
                if mol1:
                    st.image(Draw.MolToImage(mol1, size=(250, 250)), caption="Molécula 1")
                else:
                    st.warning("SMILES 1 inválido ou não reconhecido")
            
            if smile_2 and smile_2.strip():
                with cols_mol[1]:
                    mol2 = Chem.MolFromSmiles(smile_2)
                    if mol2:
                        st.image(Draw.MolToImage(mol2, size=(250, 250)), caption="Molécula 2")
                    else:
                        st.warning("SMILES 2 inválido ou não reconhecido")
    
    with tab2:
        st.subheader("Predição em Lote")
        
        st.markdown("""
        **Instruções:**  
        Faça upload de um arquivo CSV com uma das seguintes estruturas:
        
        1. **Substâncias puras:**  
           `smile_1`, `temperature`
        
        2. **Misturas:**  
           `smile_1`, `smile_2`, `fraction`, `temperature`
        
        3. **Para validação (opcional):**  
           Adicione a coluna `logV` para comparação com valores reais
        """)
        
        uploaded_file = st.file_uploader("Selecione um arquivo CSV", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
            st.write("**Pré-visualização dos dados:**")
            st.dataframe(df.head())
            
            architecture_batch = st.selectbox("Selecione o modelo", ["base", "lora"], key="batch_model")
            
            if st.button("Executar Predições em Lote", type="primary", use_container_width=True):
                with st.spinner("Processando arquivo..."):
                    resultados = predict_batch(token, df, architecture_batch)
                    
                    if "predictions" in resultados:
                        st.success(f"✅ {len(resultados['predictions'])} predições concluídas!")
                        
                        # Combinar resultados
                        df_resultados = pd.DataFrame(resultados["predictions"])
                        df_final = df.copy()
                        df_final["viscosity_pred"] = df_resultados["viscosity"]
                        
                        # Mostrar resultados
                        st.write("**Resultados:**")
                        st.dataframe(df_final)
                        
                        # Download
                        csv = df_final.to_csv(index=False)
                        st.download_button(
                            label="📥 Baixar Resultados (CSV)",
                            data=csv,
                            file_name="resultados_viscosidade.csv",
                            mime="text/csv"
                        )
                        
                        # Gráfico se houver valores reais
                        if "logV" in df_final.columns:
                            st.divider()
                            st.subheader("📊 Análise Comparativa")
                            
                            try:
                                y_real = df_final["logV"]
                                y_pred = df_final["viscosity_pred"]
                                
                                fig = plot_real_pred(y_real, y_pred)
                                st.pyplot(fig)
                                
                                r2 = r2_score(y_real, y_pred)
                                st.metric("Coeficiente de Determinação (R²)", f"{r2:.4f}")
                                
                            except Exception as e:
                                st.warning(f"Não foi possível gerar gráfico: {e}")
                    else:
                        st.error(f"Erro: {resultados.get('detail', 'Erro desconhecido')}")

# Para executar este arquivo diretamente (opcional)
if __name__ == "__main__":
    st.title("Página de Predição")
    st.warning("Execute main_app.py para usar o sistema completo.")