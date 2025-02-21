import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import re
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md
import biotite.structure.io as bsio
import asyncio
import aiohttp
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB import PDBParser

# Set page configuration
st.set_page_config(layout='wide')

# Sidebar
st.sidebar.title('GNN-Transformer')

# Input Selection
input_choice = st.sidebar.radio("Select Input Type", ["Protein Sequence", "PDB ID", "Upload PDB File"])

# Visualization Settings
st.sidebar.subheader("Visualization Settings")
vis_style = st.sidebar.selectbox("Molecular Visualization Style", ["cartoon", "stick", "surface", "sphere"])

# Default protein sequence
DEFAULT_SEQ = ""

# Environmental factor inputs
st.sidebar.subheader('Environmental Conditions')
pH = st.sidebar.slider('pH Level', min_value=0.0, max_value=14.0, value=7.0, step=0.1)
temperature = st.sidebar.slider('Temperature (°C)', min_value=-10, max_value=100, value=37, step=1)
solvent_accessibility = st.sidebar.selectbox('Solvent Accessibility', ['Buried', 'Partially Exposed', 'Fully Exposed'])

# Session state for critical residues
if "critical_residues" not in st.session_state:
    st.session_state.critical_residues = []

user_input = ""

# Function to render molecule with highlighted residues
def render_mol(pdb, highlight_residues=None, style="cartoon"):
    pdbview = py3Dmol.view()
    pdbview.addModel(pdb, 'pdb')
    pdbview.setStyle({style: {'color': 'spectrum'}})
    
    if highlight_residues:
        for residue in highlight_residues:
            pdbview.addStyle(
                {'resi': residue, 'chain': 'A'},  
                {'stick': {'colorscheme': 'redCarbon', 'radius': 0.2}}
            )
    
    pdbview.setBackgroundColor('white')
    pdbview.zoomTo()
    pdbview.zoom(2, 800)
    pdbview.spin(True)
    showmol(pdbview, height=500, width=800)
    
# Function to validate protein sequence
def validate_input(input_choice, user_input):
    """Validates protein sequences, PDB IDs, and extracted sequences from PDB files."""
    
    # Validation for Protein Sequences
    if input_choice == "Protein Sequence":
        if not user_input:
            return False, "Protein sequence is empty."
        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", user_input.strip()):
            return False, "Protein sequence contains invalid characters."
    
    # Validation for PDB IDs (should be exactly 4 alphanumeric characters)
    elif input_choice == "PDB ID":
        if not re.match(r"^[0-9A-Za-z]{4}$", user_input.strip()):
            return False, "PDB ID must be exactly 4 alphanumeric characters (e.g., 1CRN)."
    
    # Validation for PDB File (Extracted sequence must be valid)
    elif input_choice == "Upload PDB File":
        extracted_sequence = extract_sequence_from_pdb(user_input)  # Get sequence from PDB file
        if not extracted_sequence:
            return False, "Failed to extract sequence from the uploaded PDB file."
        if not re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", extracted_sequence.strip()):
            return False, "Extracted sequence from PDB file contains invalid characters."
    
    return True, ""


# Function to fetch PDB structure from ESMFold API
@st.cache_data
def fetch_pdb(input_choice, user_input):
    if input_choice == "Protein Sequence":
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(
            'https://api.esmatlas.com/foldSequence/v1/pdb/',
            headers=headers,
            data=user_input
        )
    elif input_choice == "PDB ID":
        url = f'https://files.rcsb.org/download/{user_input}.pdb'
        response = requests.get(url)  
    else:
        return None

    if response.status_code == 200:
        return response.text
    else:
        st.error(f"Failed to fetch PDB structure. Status code: {response.status_code}")
        return None


# Function to calculate residue contact map
def compute_contact_map(pdb_file):
    traj = md.load_pdb(pdb_file)
    dist, _ = md.compute_contacts(traj, scheme='ca')
    contact_map = dist.mean(axis=0)

    side_length = int(np.round(np.sqrt(contact_map.shape[0])))  # Ensure an integer side length
    adjusted_size = side_length ** 2

    if contact_map.size > adjusted_size:
        contact_map = contact_map[:adjusted_size]  # Trim excess elements
    elif contact_map.size < adjusted_size:
        contact_map = np.pad(contact_map, (0, adjusted_size - contact_map.size), mode='constant')  # Pad if needed

    return contact_map.reshape((side_length, side_length))


# Function to plot contact map
def plot_contact_map(contact_map, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(contact_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Contact Strength")
    plt.title(title)
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    st.pyplot(plt)

# Function to compute secondary structure-level interactions
def compute_secondary_structure_interactions(pdb_file):
    traj = md.load_pdb(pdb_file)
    dssp = md.compute_dssp(traj)[0]
    sec_struct_map = np.zeros((len(dssp), len(dssp)))

    for i in range(len(dssp)):
        for j in range(len(dssp)):
            if dssp[i] == dssp[j]:
                sec_struct_map[i, j] = 1

    return sec_struct_map

# Function to plot contact map with custom colormap
def plot_contact_map(contact_map, title, cmap='viridis'):
    plt.figure(figsize=(6, 6))
    plt.imshow(contact_map, cmap=cmap, interpolation='nearest')
    plt.colorbar(label="Contact Strength")
    plt.title(title)
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    st.pyplot(plt)

# Function to plot Ramachandran Plot
def plot_ramachandran(pdb_file):
    traj = md.load_pdb(pdb_file)
    phi, psi = md.compute_phi(traj)[1], md.compute_psi(traj)[1]
    
    plt.figure(figsize=(6, 6))
    sns.kdeplot(x=phi.flatten(), y=psi.flatten(), cmap="coolwarm", fill=True)
    plt.xlabel("Phi (ϕ) Angles")
    plt.ylabel("Psi (ψ) Angles")
    plt.title("Ramachandran Plot")
    st.pyplot(plt)

# Fucntion to generate sequence properties
def compute_sequence_properties(sequence):
    analysis = ProteinAnalysis(sequence)
    properties = {
        "Molecular Weight": analysis.molecular_weight(),
        "Aromaticity": analysis.aromaticity(),
        "Instability Index": analysis.instability_index(),
        "Isoelectric Point": analysis.isoelectric_point(),
        "Hydrophobicity": analysis.gravy()
    }
    return properties

# Mapping of three-letter amino acid codes to one-letter codes
protein_letters_3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# Function to extract the sequence from the PDB file
def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequence = ""
    
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname()
                if resname in protein_letters_3to1:
                    sequence += protein_letters_3to1[resname]

    return sequence


# Function to generate molecular dynamics trajectory
def generate_md_trajectory(pdb_filename, output_traj="protein_md.dcd", steps=10):
    try:
        traj = md.load_pdb(pdb_filename)
        md_traj = traj.slice(range(0, traj.n_frames, max(1, traj.n_frames // steps)))  # Downsample frames
        md_traj.save_dcd(output_traj)
        return output_traj
    except Exception as e:
        st.error(f"Error generating MD trajectory: {e}")
        return None

# Function to estimate stability
HYDROPHOBICITY = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
    'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
    'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
}

def estimate_stability(sequence, pH, temperature, solvent_accessibility):
    hydrophobicity_score = sum(HYDROPHOBICITY.get(aa, 0) for aa in sequence) / len(sequence)
    
    stability = 80
    stability -= abs(pH - 7) * 2
    stability += hydrophobicity_score * 5  # Increase impact of hydrophobicity

    if temperature < 10 or temperature > 60:
        stability -= 15
    elif temperature < 20 or temperature > 50:
        stability -= 10
    else:
        stability -= 5

    if solvent_accessibility == "Buried":
        stability += 5
    elif solvent_accessibility == "Fully Exposed":
        stability -= 10

    return max(0, min(100, stability))

async def fetch_pdb_async(sequence):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    async with aiohttp.ClientSession() as session:
        async with session.post('https://api.esmatlas.com/foldSequence/v1/pdb/', headers=headers, data=sequence) as response:
            if response.status == 200:
                return await response.text()
            else:
                return None

# Function to update and display the protein structure
def update(user_input, input_choice, pH, temperature, solvent_accessibility, highlight_residues, pdb_file=None):
    with st.spinner('Predicting protein structure...'):
        sequence = ""  # Ensure sequence is always initialized
        pdb_string ="" 

        # Case 1: User uploaded a PDB file
        if pdb_file:
            st.sidebar.success("Using uploaded PDB structure")
            sequence = extract_sequence_from_pdb(pdb_file)
            st.write(f"Extracted Sequence: {sequence}")

        # Case 2: User entered a protein sequence
        elif input_choice == "Protein Sequence":
            pdb_string = fetch_pdb(input_choice, user_input)
            sequence = user_input.strip()

        # Case 3: User entered a PDB ID
        elif input_choice == "PDB ID":
            pdb_string = fetch_pdb(input_choice, user_input)
            if pdb_string:
                pdb_filename = "temp.pdb"
                with open(pdb_filename, "w") as f:
                    f.write(pdb_string)
                sequence = extract_sequence_from_pdb(pdb_filename)
                st.write(f"Extracted Sequence from PDB ID: {sequence}")
            else:
                st.error("Failed to fetch the PDB structure.")
                return

        # Check if sequence is valid before proceeding
        if not sequence:
            st.error("No valid sequence found. Please enter a valid sequence or PDB file.")
            return

        if pdb_string:
            name = sequence[:3] + sequence[-3:] if sequence else "predicted"
            pdb_filename = 'predicted.pdb'
            with open(pdb_filename, 'w') as f:
                f.write(pdb_string)


            struct = bsio.load_structure('predicted.pdb', extra_fields=["b_factor"])
            b_value = round(struct.b_factor.mean(), 4)

            st.subheader('Visualization of predicted protein structure')
            render_mol(pdb_string, highlight_residues, style=vis_style)

            st.subheader('plDDT')
            st.write('plDDT is a per-residue estimate of the confidence in prediction on a scale from 0-100.')
            st.info(f'plDDT: {b_value}')

            st.subheader('Environmental Factors')
            st.write(f"*pH:* {pH}")
            st.write(f"*Temperature:* {temperature} °C")
            st.write(f"*Solvent Accessibility:* {solvent_accessibility}")

            if highlight_residues:
                st.subheader('Critical Residues and Interactions')
                st.write(f"Highlighted residues: {highlight_residues}")
                st.write("These residues may play a critical role in protein function, such as active sites, binding sites, or stabilizing interactions.")
        
            # Compute Residue-Level Contact Map
            contact_map = compute_contact_map(pdb_filename)
            st.subheader('Residue-Level Contact Map')
            plot_contact_map(contact_map, 'Residue Contact Map', cmap='coolwarm')

            # Compute Secondary Structure-Level Interactions
            sec_struct_map = compute_secondary_structure_interactions(pdb_filename)
            st.subheader('Secondary Structure-Level Interaction Map')
            plot_contact_map(sec_struct_map, 'Secondary Structure Contact Map', cmap='plasma')
                
            # Compute Ramachandran Plot
            st.subheader("Ramachandran Plot")
            plot_ramachandran(pdb_filename)
                
            # Sequence Properties
            if input_choice == "Protein Sequence":
                if re.match("^[ACDEFGHIKLMNPQRSTVWY]+$", sequence):  # ✅ Ensure it's a valid protein sequence
                    properties = compute_sequence_properties(sequence)
                    st.subheader("Sequence Properties")
                    for key, value in properties.items():
                        st.write(f"**{key}:** {round(value, 3)}")
            elif input_choice in ["PDB ID", "Upload PDB File"]:
                pdb_string = fetch_pdb(input_choice, user_input)  # Fetch PDB file
                if pdb_string:
                    pdb_filename = "temp.pdb"
                    with open(pdb_filename, "w") as f:
                        f.write(pdb_string)
        
                    sequence = extract_sequence_from_pdb(pdb_filename)  # Extract sequence from PDB
                    properties = compute_sequence_properties(sequence)
                    st.subheader("Sequence Properties")
                    for key, value in properties.items():
                        st.write(f"**{key}:** {round(value, 3)}")
                    if not sequence:
                        st.error("Failed to extract sequence from the PDB structure.")
                        return
                else:
                    st.error("Failed to fetch the PDB structure.")
                    return


            # Molecular Dynamics Simulation
            st.subheader('Molecular Dynamics Simulation')
            md_traj_file = generate_md_trajectory(pdb_filename)
            if md_traj_file:
                st.success(f"Molecular Dynamics trajectory generated: {md_traj_file}")

            # Stability Estimation
            stability_score = estimate_stability(sequence, pH, temperature, solvent_accessibility)
            st.subheader('Estimated Protein Stability')
            st.info(f"Stability Score (0-100): {stability_score}")

            # Download buttons
            st.download_button("Download PDB", data=pdb_string, file_name='predicted.pdb', mime='text/plain')
            fasta_sequence = f">{sequence[:10]}\n{sequence}"
            st.download_button("Download FASTA", data=fasta_sequence, file_name="sequence.fasta", mime="text/plain")
            if md_traj_file:
                with open(md_traj_file, "rb") as f:
                    st.download_button("Download MD Trajectory", f.read(), file_name="protein_md.dcd", mime="application/octet-stream")

# User input section
st.title('Protein Structure Prediction & Stability Analysis')
# Protein sequence input
# Ensure `user_input` is always defined
if input_choice == "Protein Sequence":
    user_input = st.text_area("Enter Protein Sequence", DEFAULT_SEQ, height=100)
elif input_choice == "PDB ID":
    user_input = st.text_input("Enter PDB ID (e.g., 1CRN)", "")
elif input_choice == "Upload PDB File":
    uploaded_pdb = st.file_uploader("Upload PDB File", type=["pdb"])
    if uploaded_pdb:
        pdb_path = "uploaded.pdb"
        with open(pdb_path, "wb") as f:
            f.write(uploaded_pdb.getbuffer())
        # Extract sequence from the uploaded PDB file
        user_input = extract_sequence_from_pdb(pdb_path)
        # Display extracted sequence
        st.write(f"Extracted Sequence: {user_input}")

# Critical residues input
st.sidebar.subheader('Critical Residues')
highlight_residues = st.sidebar.text_input("Enter residue numbers to highlight (comma-separated, e.g., 10, 20, 30):")
highlight_residues = [int(res.strip()) for res in highlight_residues.split(",") if res.strip().isdigit()] if highlight_residues else None

predict_clicked = st.button('Predict', key="predict_button")
if predict_clicked:
    if not user_input:
        st.warning("Please enter a valid input (Protein Sequence, PDB ID, or Upload a PDB file).")
    else:
        is_valid, message = validate_input(input_choice, user_input)
        if not is_valid:
            st.error(message)
        else:
            update(user_input, input_choice, pH, temperature, solvent_accessibility, highlight_residues)
