import streamlit as st
import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import pandas as pd

# Set page configuration
st.set_page_config(page_title="JAX Neural Network Toy Model", layout="wide")

st.title("Neural Network Toy Model")
st.markdown("""
This app visualizes a feed-forward neural network where the **output layer has no learnable weights**. 
The final prediction is simply the **sum** of all the hidden layer activations. 
""")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("1. Architecture")
input_dim = 2 
hidden_neurons = st.sidebar.slider("Hidden Layer Neurons", min_value=1, max_value=50, value=2, step=1)
activation_name = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh"])

st.sidebar.header("2. Synthetic Data")
a_slope = st.sidebar.slider("Slope 'a' (for Vector [0])", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
b_slope = st.sidebar.slider("Slope 'b' (for Vector [1])", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)
n_points = st.sidebar.slider("Number of Data Points (N)", min_value=10, max_value=1000, value=200, step=10)

st.sidebar.header("3. Noise & Outliers")
noise_sigma = st.sidebar.slider("Gaussian Noise (Sigma)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

add_outliers = st.sidebar.checkbox("Add Outliers", value=False)
n_outliers = 0
if add_outliers:
    n_outliers = st.sidebar.slider("Number of Outliers", min_value=1, max_value=20, value=5, step=1)

st.sidebar.header("4. Training Parameters")
lr = st.sidebar.number_input("Learning Rate", min_value=0.001, max_value=1.0, value=0.1, step=0.01, format="%.3f")
epochs = st.sidebar.slider("Training Steps", min_value=10, max_value=10000, value=1000, step=50)

st.sidebar.header("5. Regularization")
reg_type = st.sidebar.selectbox("Regularization Type", ["None", "L1", "L2", "Inverse L1", "Inverse L2", "Log"])
reg_lambda = st.sidebar.number_input("Regularization Strength (Lambda)", min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("By [@ineporozhnii (GitHub Page)](https://github.com/ineporozhnii/neural-network-toy-model)", unsafe_allow_html=True)

# ==========================================
# DATA GENERATION (Fully Vectorized)
# ==========================================
def relu(x):
    return jnp.maximum(0.0, x)

def tanh(x):
    return jnp.tanh(x)

act_fn = relu if activation_name == "ReLU" else tanh

def true_function(X_matrix):
    slopes = jnp.array([a_slope, b_slope])
    return jnp.sum(act_fn(X_matrix * slopes), axis=1, keepdims=True)

# 1. Random uniformly sampled data points
np.random.seed(42)
X_train_np = np.random.uniform(-1, 1, size=(n_points, input_dim))
y_train_clean = np.array(true_function(X_train_np))

# 2. Add Gaussian Noise
noise = np.random.normal(0, noise_sigma, size=(n_points, 1))
y_train_np = y_train_clean + noise

# 3. Add Outliers (> 3 sigma)
outlier_indices = []
if add_outliers and n_outliers > 0:
    actual_outliers = min(n_outliers, n_points)
    outlier_indices = np.random.choice(n_points, actual_outliers, replace=False)
    
    outlier_magnitude = (3.1 * noise_sigma) + np.random.uniform(1.0, 2.5, size=(actual_outliers, 1))
    signs = np.random.choice([-1, 1], size=(actual_outliers, 1))
    
    y_train_np[outlier_indices] += signs * outlier_magnitude

# 4. High-resolution grid for the 3D surface (Used as pure Validation Data)
x_grid = np.linspace(-1, 1, 50)
y_grid = np.linspace(-1, 1, 50)
xx, yy = np.meshgrid(x_grid, y_grid)

grid_points = np.column_stack((xx.ravel(), yy.ravel()))
zz_true = np.array(true_function(grid_points)).reshape(50, 50)

# Validation set setup
X_val_np = grid_points
y_val_np = zz_true.reshape(-1, 1)

# ==========================================
# VISUALIZE DATA
# ==========================================
st.subheader("Synthetic Dataset Visualization")

fig = go.Figure()

fig.add_trace(go.Surface(z=zz_true, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.6, name='True Surface', showscale=True))

normal_mask = np.ones(n_points, dtype=bool)
normal_mask[outlier_indices] = False

fig.add_trace(go.Scatter3d(
    x=X_train_np[normal_mask, 0], 
    y=X_train_np[normal_mask, 1], 
    z=y_train_np[normal_mask].flatten(), 
    mode='markers',
    marker=dict(size=4, color='dodgerblue', line=dict(width=1, color='darkblue')),
    name='Clean Data / Noise'
))

if len(outlier_indices) > 0:
    fig.add_trace(go.Scatter3d(
        x=X_train_np[outlier_indices, 0], 
        y=X_train_np[outlier_indices, 1], 
        z=y_train_np[outlier_indices].flatten(), 
        mode='markers',
        marker=dict(size=7, color='red', symbol='cross', line=dict(width=2, color='darkred')),
        name='Outliers'
    ))

fig.update_layout(
    scene=dict(xaxis_title='Input Vector [0]', yaxis_title='Input Vector [1]', zaxis_title='Target Z'),
    margin=dict(l=0, r=0, b=0, t=30),
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# ==========================================
# JAX MODEL IMPLEMENTATION (Summation Output)
# ==========================================
def init_params(key, in_dim, hid_dim):
    W1 = jax.random.normal(key, (in_dim, hid_dim)) * jnp.sqrt(2.0 / in_dim) 
    return {"W1": W1}

def forward(params, X):
    hidden = act_fn(jnp.dot(X, params["W1"]))
    output = jnp.sum(hidden, axis=1, keepdims=True)
    return output

# Pure MSE Loss (Used for validation tracking, no penalty)
def mse_loss_fn(params, X, y):
    preds = forward(params, X)
    return jnp.mean((preds - y) ** 2)

# Training Loss (Includes Regularization Penalty)
def create_training_loss_fn(selected_reg):
    def training_loss_fn(params, X, y, reg_strength):
        mse = mse_loss_fn(params, X, y)
        w = params["W1"]
        eps = 1e-6  # small epsilon to prevent division by zero in inverse penalties
        
        if selected_reg == "L1":
            penalty = reg_strength * jnp.sum(jnp.abs(w))
        elif selected_reg == "L2":
            penalty = reg_strength * jnp.sum(w ** 2)
        elif selected_reg == "Inverse L1":
            penalty = reg_strength * jnp.sum(1.0 / (jnp.abs(w) + eps))
        elif selected_reg == "Inverse L2":
            penalty = reg_strength * jnp.sum(1.0 / (w ** 2 + eps))
        elif selected_reg == "Log":
            penalty = reg_strength * jnp.sum(jnp.log(jnp.abs(w) + 1.0))
        else:
            penalty = 0.0
            
        return mse + penalty
    return training_loss_fn

# Define the specific loss and update function based on user's current selection
current_loss_fn = create_training_loss_fn(reg_type)

@jax.jit
def update(params, X, y, learning_rate, reg_strength):
    loss, grads = jax.value_and_grad(current_loss_fn)(params, X, y, reg_strength)
    new_params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params, grads)
    return new_params, loss

# ==========================================
# TRAINING EXECUTION & OUTCOMES
# ==========================================
st.divider()
st.subheader("Model Training")

if st.button("🚀 Train Model", type="primary"):
    key = jax.random.PRNGKey(42)
    params = init_params(key, input_dim, hidden_neurons)
    
    X_jnp = jnp.array(X_train_np)
    y_jnp = jnp.array(y_train_np)
    X_val_jnp = jnp.array(X_val_np)
    y_val_jnp = jnp.array(y_val_np)

    loss_history = []
    val_loss_history = []
    w1_history = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(epochs):
        # Update on training data (includes regularization)
        params, train_loss = update(params, X_jnp, y_jnp, lr, reg_lambda)
        
        # Calculate pure validation loss (MSE only, no regularization penalty)
        val_loss = mse_loss_fn(params, X_val_jnp, y_val_jnp)
        
        # Handle potential NaNs from exploding inverse penalties
        if jnp.isnan(train_loss):
            st.error("Training diverged! The weights likely exploded to infinity due to a strong Inverse Penalty or high Learning Rate.")
            break
            
        loss_history.append(float(train_loss))
        val_loss_history.append(float(val_loss))
        
        if step % max(1, epochs // 100) == 0 or step == epochs - 1:
            w1_history.append(np.array(params["W1"]).flatten())
            progress_bar.progress((step + 1) / epochs)
            status_text.text(f"Training Step: {step+1}/{epochs} | Train Loss (MSE+Reg): {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    progress_bar.empty()
    status_text.success(f"Training Complete! Final Train Loss: {loss_history[-1]:.4f} | Final Val MSE: {val_loss_history[-1]:.4f}")

    # --- Print Weight Matrices ---
    with st.expander("🔍 View Network Schematic & Final Weight Matrices", expanded=True):
        st.markdown("### Network Schematic")
        st.info(f"**Input Vector** (Shape: `N x 2`) ➔ **Hidden Layer** ({hidden_neurons} Neurons) ➔ **Summation Node**")
        
        reg_math = ""
        if reg_type == "L1":
            reg_math = f" + {reg_lambda} \cdot \sum |W_1|"
        elif reg_type == "L2":
            reg_math = f" + {reg_lambda} \cdot \sum W_1^2"
        elif reg_type == "Inverse L1":
            reg_math = f" + {reg_lambda} \cdot \sum \\frac{{1}}{{|W_1| + \\epsilon}}"
        elif reg_type == "Inverse L2":
            reg_math = f" + {reg_lambda} \cdot \sum \\frac{{1}}{{W_1^2 + \\epsilon}}"
        elif reg_type == "Log":
            reg_math = f" + {reg_lambda} \cdot \sum \\log(|W_1| + 1)"
            
        st.markdown(f"**Loss Function**: $\\text{{MSE}}(y, \\hat{{y}}){reg_math}$")
        
        st.write(f"#### Final $W_1$ (Input ➔ Hidden)")
        w1_df_final = pd.DataFrame(
            np.array(params["W1"]), 
            index=["Vector [0]", "Vector [1]"], 
            columns=[f"Hidden {i+1}" for i in range(hidden_neurons)]
        )
        st.dataframe(w1_df_final)

    # --- Plotting Training Dynamics ---
    st.markdown("### Training Dynamics")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Loss Over Time (Log Scale)**")
        
        fig_loss = go.Figure()
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(loss_history))), 
            y=loss_history, 
            mode='lines', 
            name=f'Train Loss ({reg_type})',
            line=dict(color='dodgerblue', width=2)
        ))
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(len(val_loss_history))), 
            y=val_loss_history, 
            mode='lines', 
            name='Validation MSE',
            line=dict(color='darkorange', width=2, dash='dash')
        ))
        
        fig_loss.update_layout(
            yaxis_type="log",
            yaxis_title="Loss (Log Scale)",
            xaxis_title="Training Steps",
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    with col_b:
        st.markdown("**$W_1$ (Input ➔ Hidden) Evolution**")
        w1_cols = [f"Vec_[{i}] ➔ Hid_{j+1}" for i in range(input_dim) for j in range(hidden_neurons)]
        st.line_chart(pd.DataFrame(w1_history, columns=w1_cols))

    # --- Plotting 2D Vector Representation of Hidden Neurons ---
    st.markdown("### Hidden Neuron Weights as 2D Vectors")
    
    fig_vec = go.Figure()
    
    w1_x = np.array(params["W1"][0, :])
    w1_y = np.array(params["W1"][1, :])
    
    if hidden_neurons == 1:
        vector_colors = ['dodgerblue']
    else:
        vector_colors = pcolors.sample_colorscale('Turbo', [i / (hidden_neurons - 1) for i in range(hidden_neurons)])
    
    fig_vec.add_trace(go.Scatter(
        x=w1_x, y=w1_y, 
        mode='markers+text',
        text=[f"N{j+1}" for j in range(hidden_neurons)],
        textposition='top center',
        marker=dict(size=8, color=vector_colors),
        name="Neuron Vectors"
    ))

    for j in range(hidden_neurons):
        fig_vec.add_annotation(
            ax=0, ay=0, 
            x=w1_x[j], y=w1_y[j], 
            xref='x', yref='y', axref='x', ayref='y',
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5,
            arrowcolor=vector_colors[j]
        )

    fig_vec.add_trace(go.Scatter(
        x=[0], y=[0], mode='markers', 
        marker=dict(color='black', size=10, symbol='cross'), 
        name='Origin (0,0)'
    ))

    # Calculate symmetrical axis limits to center origin visually
    max_val = max(np.max(np.abs(w1_x)), np.max(np.abs(w1_y)), 1.0) * 1.2
    
    fig_vec.update_layout(
        xaxis=dict(range=[-max_val, max_val], zeroline=True, zerolinewidth=2, zerolinecolor='lightgray', title="Weight for Vector [0]"),
        yaxis=dict(range=[-max_val, max_val], zeroline=True, zerolinewidth=2, zerolinecolor='lightgray', title="Weight for Vector [1]", scaleanchor="x", scaleratio=1),
        height=500,
        showlegend=False,
        plot_bgcolor='white'
    )
    fig_vec.update_xaxes(showgrid=True, gridwidth=1, gridcolor='whitesmoke')
    fig_vec.update_yaxes(showgrid=True, gridwidth=1, gridcolor='whitesmoke')
    
    st.plotly_chart(fig_vec, use_container_width=True)

    # --- Plotting Final Prediction vs True Surface ---
    st.subheader("Final Outcome: Model Prediction vs True Surface")
    
    preds_grid = np.array(forward(params, jnp.array(grid_points))).reshape(xx.shape)

    fig2 = go.Figure()
    
    fig2.add_trace(go.Surface(
        z=zz_true, x=x_grid, y=y_grid, 
        colorscale='Blues', opacity=0.4, 
        name='True Target', showscale=False
    ))
    
    fig2.add_trace(go.Surface(
        z=preds_grid, x=x_grid, y=y_grid, 
        colorscale='Inferno', opacity=0.9, 
        name='Model Prediction'
    ))

    fig2.update_layout(
        scene=dict(xaxis_title='Input Vector [0]', yaxis_title='Input Vector [1]', zaxis_title='Target Z'),
        title="Comparison: Predicted Surface (Inferno) vs. True Target (Translucent Blue)",
        height=600
    )
    st.plotly_chart(fig2, use_container_width=True)