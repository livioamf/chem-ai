import streamlit as st
import requests

API_BASE = "http://localhost:8010"

# Configuração da página
st.set_page_config(
    page_title="Sistema de Viscosidade",
    layout="wide",
    page_icon="🧪",
    initial_sidebar_state="expanded"
)

# Função de login
def login(username, password):
    data = {"username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    try:
        resp = requests.post(f"{API_BASE}/auth/token", data=data, headers=headers)
        if resp.status_code != 200:
            return None
        
        token = resp.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {token}"}
        user_info = requests.get(f"{API_BASE}/users/me", headers=auth_headers)
        
        if user_info.status_code == 200:
            user_data = user_info.json()
            user_type = user_data.get("type", "user")
            return {
                "access_token": token,
                "user_type": user_type,
                "username": username,
                "user_data": user_data
            }
        
        return {"access_token": token, "user_type": "user", "username": username}
        
    except Exception:
        return None

# Inicializar estado da sessão
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_page" not in st.session_state:
    st.session_state.current_page = "predicao"  # Página padrão
if "user_info" not in st.session_state:
    st.session_state.user_info = {}

# Funções de navegação
def go_to_predicao():
    st.session_state.current_page = "predicao"

def go_to_admin():
    st.session_state.current_page = "admin"

def logout():
    st.session_state.logged_in = False
    st.session_state.user_info = {}
    st.session_state.current_page = "predicao"

# Sidebar para login/navegação
with st.sidebar:
    st.title("🔐 Login & Navegação")
    
    if not st.session_state.logged_in:
        st.subheader("Entrar no Sistema")
        username = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        
        if st.button("Entrar", use_container_width=True):
            if username and password:
                user_data = login(username, password)
                if user_data:
                    st.session_state.logged_in = True
                    st.session_state.user_info = user_data
                    st.session_state.current_page = "predicao"
                    st.success(f"Bem-vindo, {username}!")
                    st.rerun()
                else:
                    st.error("Usuário ou senha incorretos")
            else:
                st.warning("Preencha todos os campos")
    
    else:
        st.subheader(f"👤 {st.session_state.user_info.get('username', 'Usuário')}")
        st.caption(f"Tipo: {st.session_state.user_info.get('user_type', 'user')}")
        
        st.divider()
        st.subheader("🧭 Navegação")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧪 Predição", use_container_width=True, 
                        type="primary" if st.session_state.current_page == "predicao" else "secondary"):
                go_to_predicao()
                st.rerun()
        
        with col2:
            if st.session_state.user_info.get("user_type") == "Administrador":
                if st.button("👥 Admin", use_container_width=True,
                           type="primary" if st.session_state.current_page == "admin" else "secondary"):
                    go_to_admin()
                    st.rerun()
            else:
                st.button("👥 Admin", disabled=True, use_container_width=True, 
                         help="Apenas administradores")
        
        st.divider()
        
        if st.button("🚪 Sair", use_container_width=True, type="secondary"):
            logout()
            st.rerun()

st.title("🧪 Sistema de Predição de Viscosidade")

if not st.session_state.logged_in:
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bem-vindo ao Sistema")
        st.markdown("""
        Este sistema permite:
        
        🔬 **Predição de Viscosidade**
        - Predição individual
        - Predição em lote via CSV
        - Visualização de moléculas
        - Gráficos comparativos
        
        👥 **Administração** (apenas para administradores)
        - Gerenciamento de usuários
        - Controle de acesso
        
        **Faça login na barra lateral para começar.**
        """)
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/1998/1998678.png", width=300)
    
else:
    st.markdown(f"**Página atual:** {st.session_state.current_page.upper()}")
    st.divider()
    
    if st.session_state.current_page == "predicao":
        try:
            import chemapp
            chemapp.show_predicao_page(st.session_state.user_info["access_token"])
        except ImportError:
            st.warning("Módulo de predição não encontrado. Certifique-se de que predicao.py está na mesma pasta.")
            st.info("Esta seria a página de predição de viscosidade.")
            
            if st.button("Exemplo de predição"):
                st.success("Funcionalidade de predição carregada!")
    
    elif st.session_state.current_page == "admin":
        if st.session_state.user_info.get("user_type") == "Administrador":
            try:
                import app_admin
                app_admin.show_admin_page(st.session_state.user_info["access_token"])
            except ImportError:
                st.warning("Módulo de administração não encontrado. Certifique-se de que admin.py está na mesma pasta.")
                st.info("Esta seria a página de administração de usuários.")
        else:
            st.error("⛔ Acesso negado. Esta área é restrita a administradores.")
            st.info("Volte para a página de predição usando o menu de navegação.")