import streamlit as st
import pandas as pd
import requests

API_BASE = "http://localhost:8010"

def get_auth_headers(token):
    return {"Authorization": f"Bearer {token}"}

def get_users(token):
    resp = requests.get(f"{API_BASE}/users", headers=get_auth_headers(token))
    if resp.status_code == 200:
        return resp.json().get("users", [])
    else:
        st.error(f"Erro ao buscar usuários: {resp.status_code}")
        return []

def create_user(token, name, username, password, user_type):
    data = {
        "name": name,
        "username": username,
        "password": password,
        "type": user_type
    }
    resp = requests.post(f"{API_BASE}/users", json=data, headers=get_auth_headers(token))
    return resp

def update_user(token, user_id, name, username, password, user_type):
    data = {
        "name": name,
        "username": username,
        "password": password,
        "type": user_type
    }
    resp = requests.put(f"{API_BASE}/users/{user_id}", json=data, headers=get_auth_headers(token))
    return resp

def delete_user(token, user_id):
    resp = requests.delete(f"{API_BASE}/users/{user_id}", headers=get_auth_headers(token))
    return resp

def show_admin_page(token):
    """Função principal da página de administração"""
    
    st.header("👥 Administração de Usuários")
    st.markdown("Gerencie os usuários do sistema.")
    
    # Abas para diferentes funcionalidades
    tab1, tab2, tab3 = st.tabs(["📋 Listar Usuários", "➕ Cadastrar Novo", "⚙️ Editar/Excluir"])
    
    with tab1:
        st.subheader("Usuários do Sistema")
        
        users = get_users(token)
        if users:
            # Criar DataFrame
            df = pd.DataFrame(users)
            
            # Mostrar tabela
            st.dataframe(
                df[['id', 'name', 'username', 'type']],
                use_container_width=True,
                hide_index=True
            )
            
            # Estatísticas
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total de Usuários", len(users))
            with col2:
                admin_count = sum(1 for u in users if u.get('type') == 'Administrador')
                st.metric("Administradores", admin_count)
        else:
            st.info("Nenhum usuário cadastrado no sistema.")
    
    with tab2:
        st.subheader("Cadastrar Novo Usuário")
        
        with st.form("form_cadastro", clear_on_submit=True):
            col1, col2 = st.columns(2)
            
            with col1:
                nome = st.text_input("Nome completo*")
                usuario = st.text_input("Nome de usuário*")
            
            with col2:
                senha = st.text_input("Senha*", type="password")
                tipo = st.selectbox("Tipo de usuário*", ["Usuário", "Administrador"])
            
            submitted = st.form_submit_button("Cadastrar Usuário", use_container_width=True)
            
            if submitted:
                if not all([nome, usuario, senha]):
                    st.error("Preencha todos os campos obrigatórios (*)")
                else:
                    resp = create_user(token, nome, usuario, senha, tipo)
                    if resp.status_code == 201:
                        st.success(f"Usuário '{usuario}' criado com sucesso!")
                        st.rerun()
                    else:
                        error_msg = resp.json().get('detail', 'Erro desconhecido')
                        st.error(f"Erro ao criar usuário: {error_msg}")
    
    with tab3:
        st.subheader("Gerenciar Usuários Existentes")
        
        users = get_users(token)
        if not users:
            st.info("Nenhum usuário para gerenciar.")
        else:
            # Criar lista para seleção
            user_options = []
            for user in users:
                display_name = f"{user['name']} ({user['username']})"
                user_options.append({
                    "display": display_name,
                    "id": user['id'],
                    "data": user
                })
            
            # Selecionar usuário
            selected_display = st.selectbox(
                "Selecione um usuário:",
                [f"{u['data']['username']} - {u['data']['name']}" for u in user_options]
            )
            
            # Encontrar usuário selecionado
            selected_user = None
            for u in user_options:
                if f"{u['data']['username']} - {u['data']['name']}" == selected_display:
                    selected_user = u
                    break
            
            if selected_user:
                st.divider()
                st.write(f"**Editando:** {selected_user['data']['name']}")
                
                # Formulário de edição
                with st.form("form_edicao"):
                    novo_nome = st.text_input("Nome", value=selected_user['data']['name'])
                    novo_usuario = st.text_input("Usuário", value=selected_user['data']['username'])
                    nova_senha = st.text_input(
                        "Nova senha (deixe em branco para manter a atual)",
                        type="password",
                        placeholder="Digite apenas se quiser alterar"
                    )
                    novo_tipo = st.selectbox(
                        "Tipo",
                        ["Administrador", "Usuário"],
                        index=0 if selected_user['data']['type'] == 'Administrador' else 1
                    )
                    
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        atualizar = st.form_submit_button("🔄 Atualizar", use_container_width=True)
                    
                    if atualizar:
                        if not novo_nome or not novo_usuario:
                            st.error("Nome e usuário são obrigatórios")
                        else:
                            resp = update_user(
                                token,
                                selected_user['id'],
                                novo_nome,
                                novo_usuario,
                                nova_senha if nova_senha else None,
                                novo_tipo
                            )
                            if resp.status_code == 200:
                                st.success("Usuário atualizado com sucesso!")
                                st.rerun()
                            else:
                                error_msg = resp.json().get('detail', 'Erro desconhecido')
                                st.error(f"Erro ao atualizar: {error_msg}")
                
                # Área de exclusão
                st.divider()
                st.subheader("Exclusão de Usuário")
                
                with st.expander("⚠️ Área perigosa - Clique para expandir"):
                    st.warning(f"Esta ação irá remover permanentemente o usuário **{selected_user['data']['username']}**.")
                    
                    confirm = st.checkbox(f"Confirmo que desejo excluir {selected_user['data']['username']}")
                    
                    if confirm:
                        if st.button("🗑️ Excluir Usuário", type="secondary", use_container_width=True):
                            resp = delete_user(token, selected_user['id'])
                            if resp.status_code == 200:
                                st.success("Usuário excluído com sucesso!")
                                st.rerun()
                            else:
                                error_msg = resp.json().get('detail', 'Erro desconhecido')
                                st.error(f"Erro ao excluir: {error_msg}")

# Para executar este arquivo diretamente (opcional)
if __name__ == "__main__":
    st.title("Página de Administração")
    st.warning("Execute main_app.py para usar o sistema completo.")