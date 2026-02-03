import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy import linalg, stats, signal
import xarray as xr
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Enhanced SOTA Fracture Model
# ============================================================================

class EnhancedFractureModel:
    """
    Enhanced fracture modeling system combining best features from both codes
    Features:
    1. Orthorhombic anisotropy with HTI/VTI combinations
    2. Stress sensitivity via Third-Order Elasticity
    3. Frequency-dependent response
    4. Brown-Korringa fluid substitution for anisotropic media
    5. Advanced synthetic seismic generation
    6. Bayesian inversion framework
    """
    
    def __init__(self,
                 model_type: str = 'orthorhombic',
                 use_geomechanics: bool = True,
                 frequency_dependent: bool = False,
                 use_fluid_substitution: bool = True):
        """
        Initialize enhanced fracture model
        
        Parameters:
        -----------
        model_type : str
            'orthorhombic', 'HTI', 'VTI_HTI', or 'monoclinic'
        use_geomechanics : bool
            Include stress sensitivity using Third-Order Elasticity
        frequency_dependent : bool
            Include frequency-dependent fracture response
        use_fluid_substitution : bool
            Include Brown-Korringa fluid substitution
        """
        self.model_type = model_type
        self.use_geomechanics = use_geomechanics
        self.frequency_dependent = frequency_dependent
        self.use_fluid_substitution = use_fluid_substitution
        
        # Physical constants
        self.fluid_properties = {
            'brine': {'rho': 1040, 'K': 2.8e9, 'mu': 0.001},
            'oil': {'rho': 800, 'K': 1.0e9, 'mu': 0.005},
            'gas': {'rho': 200, 'K': 0.1e9, 'mu': 0.0001}
        }
        
        # Mineral properties
        self.mineral_properties = {
            'quartz': {'K': 37e9, 'G': 44e9, 'rho': 2650},
            'calcite': {'K': 70e9, 'G': 32e9, 'rho': 2710},
            'clay': {'K': 25e9, 'G': 15e9, 'rho': 2600}
        }
        
        # Initialize modules
        if self.use_geomechanics:
            self.third_order_constants = self._initialize_toe_constants()
            
        if self.frequency_dependent:
            self.dispersion_params = self._initialize_dispersion_model()
    
    def _initialize_toe_constants(self) -> Dict:
        """Initialize Third-Order Elasticity constants"""
        return {
            'c111': -18000,  'c112': -6000,
            'c123': -4000,   'c144': -1500,
            'c155': -3000,   'c456': -1000
        }
    
    def _initialize_dispersion_model(self) -> Dict:
        """Initialize frequency-dependent fracture model parameters"""
        return {
            'characteristic_frequency': 50,  # Hz
            'fluid_mobility': 1e-12,  # m^2/Pa·s
            'fracture_scale': 0.1,  # m
            'Q_min': 20,  'Q_max': 100,
            'squirt_flow': True,
            'meso_scale_effects': True
        }
    
    def effective_medium_model(self,
                              background: Dict,
                              fractures: Dict,
                              stress: Optional[Dict] = None,
                              fluid_props: Optional[Dict] = None) -> Dict:
        """
        Calculate effective elastic stiffness with enhanced physics
        
        Parameters:
        -----------
        background : Dict with rock properties
        fractures : Dict with fracture properties (can include multiple sets)
        stress : Optional stress field
        fluid_props : Optional fluid properties for substitution
        
        Returns:
        --------
        Dict with stiffness tensor and derived properties
        """
        
        # Base stiffness from background rock
        C_bg = self._background_stiffness(background)
        
        # Add fracture effects (can handle multiple fracture sets)
        if 'fracture_sets' in fractures:
            # Multiple fracture sets
            for frac_set in fractures['fracture_sets']:
                C_bg = self._add_fracture_set(C_bg, background, frac_set)
        else:
            # Single fracture set
            C_bg = self._add_fracture_set(C_bg, background, fractures)
        
        # Apply stress sensitivity
        if self.use_geomechanics and stress is not None:
            C_bg = self._apply_stress_sensitivity(C_bg, stress, background)
        
        # Apply fluid substitution if requested
        if self.use_fluid_substitution and fluid_props is not None:
            C_bg = self._brown_korringa_substitution(C_bg, background, fluid_props)
        
        # Calculate effective properties
        rho_eff = self._calculate_effective_density(background, fractures)
        
        # Extract anisotropy parameters
        anisotropy_params = self._calculate_anisotropy_parameters(C_bg, rho_eff)
        
        # Apply frequency dispersion if requested
        if self.frequency_dependent:
            C_bg = self._apply_frequency_dispersion(C_bg, background, fractures)
        
        return {
            'stiffness': C_bg,
            'density': rho_eff,
            'anisotropy_params': anisotropy_params,
            'background': background,
            'fractures': fractures
        }
    
    def _background_stiffness(self, background: Dict) -> np.ndarray:
        """Calculate background rock stiffness"""
        Vp_bg = background['Vp']
        Vs_bg = background['Vs']
        rho_bg = background['rho']
        
        mu_bg = rho_bg * Vs_bg**2
        lambda_bg = rho_bg * Vp_bg**2 - 2 * mu_bg
        
        C_bg = np.zeros((6, 6))
        C_bg[0:3, 0:3] = lambda_bg
        np.fill_diagonal(C_bg, mu_bg)
        C_bg[0, 0] = C_bg[1, 1] = C_bg[2, 2] = lambda_bg + 2*mu_bg
        
        # Add intrinsic anisotropy if present
        if 'epsilon' in background and 'delta' in background:
            C_bg = self._add_intrinsic_anisotropy(C_bg, background)
        
        return C_bg
    
    def _add_intrinsic_anisotropy(self, C: np.ndarray, background: Dict) -> np.ndarray:
        """Add intrinsic VTI anisotropy to stiffness tensor"""
        epsilon = background.get('epsilon', 0.0)
        delta = background.get('delta', 0.0)
        gamma = background.get('gamma', 0.0)
        
        C_ani = C.copy()
        
        # Apply Thomsen parameters
        Vp0 = np.sqrt(C[2, 2] / background['rho'])
        Vs0 = np.sqrt(C[5, 5] / background['rho'])
        
        # Update for VTI
        C_ani[0, 0] = C_ani[1, 1] = background['rho'] * Vp0**2 * (1 + 2 * epsilon)
        C_ani[2, 2] = background['rho'] * Vp0**2
        C_ani[0, 2] = C_ani[1, 2] = background['rho'] * (Vp0**2 - 2 * Vs0**2) * (1 + delta)
        C_ani[5, 5] = background['rho'] * Vs0**2 * (1 + 2 * gamma)
        
        return C_ani
    
    def _add_fracture_set(self, C_bg: np.ndarray, background: Dict, 
                         fractures: Dict) -> np.ndarray:
        """Add fracture compliance using Schoenberg linear slip theory"""
        
        fracture_density = fractures['density']
        aspect_ratio = fractures.get('aspect_ratio', 0.01)
        orientation = fractures.get('orientation', 0.0)
        fill = fractures.get('fill', 'fluid')
        
        # Background moduli
        mu_bg = background['rho'] * background['Vs']**2
        lambda_bg = background['rho'] * background['Vp']**2 - 2 * mu_bg
        K_bg = lambda_bg + 2/3 * mu_bg
        
        # Fracture compliances based on fill type
        if fill in self.fluid_properties:
            fluid = self.fluid_properties[fill]
            K_f = fluid['K']
            
            # Hudson model for fluid-filled fractures
            ZN = 4 * aspect_ratio / (3 * K_f + 4 * mu_bg)
            ZT = 16 * aspect_ratio / (3 * (3 * K_f + 4 * mu_bg))
        else:
            # Dry or mineral-filled fractures
            ZN = aspect_ratio / mu_bg
            ZT = 2 * ZN  # Approximate
        
        # Scale by fracture density
        BN = fracture_density * ZN
        BT = fracture_density * ZT
        
        # Fracture compliance tensor in fracture coordinates
        S_frac = np.zeros((6, 6))
        S_frac[2, 2] = BN  # Normal compliance
        S_frac[3, 3] = S_frac[4, 4] = BT  # Tangential compliances
        
        # Rotate to global coordinates
        R = self._rotation_matrix_voigt(orientation, dip=fractures.get('dip', 0))
        S_frac_global = R.T @ S_frac @ R
        
        # Effective compliance
        S_bg = linalg.inv(C_bg)
        S_eff = S_bg + S_frac_global
        
        return linalg.inv(S_eff)
    
    def _rotation_matrix_voigt(self, azimuth: float, dip: float = 0) -> np.ndarray:
        """Create rotation matrix for stiffness tensor in Voigt notation"""
        cos_a = np.cos(azimuth)
        sin_a = np.sin(azimuth)
        cos_d = np.cos(dip)
        sin_d = np.sin(dip)
        
        R_3x3 = np.array([
            [cos_a*cos_d, -sin_a, cos_a*sin_d],
            [sin_a*cos_d, cos_a, sin_a*sin_d],
            [-sin_d, 0, cos_d]
        ])
        
        # Convert to Voigt notation (6x6)
        M = np.zeros((6, 6))
        
        for i in range(3):
            for j in range(3):
                M[i, j] = R_3x3[i, j]**2
                M[i, j+3] = 2 * R_3x3[i, (j+1)%3] * R_3x3[i, (j+2)%3]
        
        for i in range(3):
            for j in range(3):
                M[i+3, j] = R_3x3[(i+1)%3, j] * R_3x3[(i+2)%3, j]
                M[i+3, j+3] = (R_3x3[(i+1)%3, (j+1)%3] * R_3x3[(i+2)%3, (j+2)%3] +
                              R_3x3[(i+1)%3, (j+2)%3] * R_3x3[(i+2)%3, (j+1)%3])
        
        return M
    
    def _apply_stress_sensitivity(self, C: np.ndarray, stress: Dict, 
                                 background: Dict) -> np.ndarray:
        """Apply stress-induced anisotropy using Third-Order Elasticity"""
        sigma_v = stress['vertical']
        sigma_h = stress['horizontal']
        sigma_H = stress.get('horizontal_max', sigma_h)
        
        # Calculate stress-induced changes
        c = self.third_order_constants
        
        # Strain from stress
        S_iso = linalg.inv(C)
        epsilon = S_iso @ np.array([sigma_H, sigma_h, sigma_v, 0, 0, 0])
        
        # Third-order corrections (simplified)
        delta_C = np.zeros((6, 6))
        
        # For orthorhombic symmetry
        delta_epsilon1 = c['c111'] * epsilon[0]**2 + c['c112'] * (epsilon[1]**2 + epsilon[2]**2)
        delta_epsilon2 = c['c111'] * epsilon[1]**2 + c['c112'] * (epsilon[0]**2 + epsilon[2]**2)
        delta_epsilon3 = c['c111'] * epsilon[2]**2 + c['c112'] * (epsilon[0]**2 + epsilon[1]**2)
        
        delta_C[0, 0] = delta_epsilon1
        delta_C[1, 1] = delta_epsilon2
        delta_C[2, 2] = delta_epsilon3
        
        # Shear terms
        delta_C[3, 3] = c['c144'] * epsilon[2]  # Simplified
        delta_C[4, 4] = c['c155'] * epsilon[1]
        delta_C[5, 5] = c['c155'] * epsilon[0]
        
        return C + delta_C
    
    def _brown_korringa_substitution(self, C: np.ndarray, background: Dict,
                                    fluid_props: Dict) -> np.ndarray:
        """Brown-Korringa fluid substitution for anisotropic media"""
        
        # Extract parameters
        phi = fluid_props.get('porosity', 0.2)
        K_s = fluid_props.get('mineral_K', 37e9)  # Mineral bulk modulus
        G_s = fluid_props.get('mineral_G', 44e9)  # Mineral shear modulus
        K_f = fluid_props.get('fluid_K', 2.2e9)   # Fluid bulk modulus
        
        # Current moduli from stiffness
        rho = background['rho']
        Vp = np.sqrt(C[2, 2] / rho)
        Vs = np.sqrt(C[5, 5] / rho)
        
        K_dry = rho * (Vp**2 - 4/3 * Vs**2)  # Dry bulk modulus
        G_dry = rho * Vs**2  # Dry shear modulus
        
        # Brown-Korringa equations for anisotropic media
        # Simplified version - full implementation requires more complex tensor math
        beta = 1 - (K_s / K_dry)
        
        # Saturated bulk modulus
        K_sat = K_s + (beta**2) / ((phi / K_f) + ((beta - phi) / K_dry) - 
                                  (background.get('delta', 0) * K_s) / (3 * K_dry))
        
        # Saturated shear modulus (less affected by fluid)
        G_sat = G_dry * (1 - background.get('gamma', 0) * K_s / (3 * K_dry))
        
        # Update stiffness components
        C_sat = C.copy()
        lambda_sat = K_sat - 2/3 * G_sat
        
        # Update diagonal components
        C_sat[0, 0] = C_sat[1, 1] = C_sat[2, 2] = lambda_sat + 2 * G_sat
        C_sat[3, 3] = C_sat[4, 4] = C_sat[5, 5] = G_sat
        
        # Update density
        new_density = background['rho'] * (1 - phi) + fluid_props.get('fluid_density', 1000) * phi
        
        return C_sat
    
    def _apply_frequency_dispersion(self, C_static: np.ndarray,
                                   background: Dict, fractures: Dict) -> np.ndarray:
        """Apply frequency-dependent stiffness using Chapman's model"""
        
        if not self.frequency_dependent:
            return C_static
        
        f_char = self.dispersion_params['characteristic_frequency']
        omega = 2 * np.pi * self.dispersion_params.get('frequency', 30)
        
        # Complex stiffness for viscoelastic effects
        Q = self.dispersion_params.get('Q', 50)
        alpha = 1 / (np.pi * Q)  # Attenuation coefficient
        
        # Frequency-dependent correction
        f_ratio = omega / (2 * np.pi * f_char)
        dispersion_factor = 1 + 0.1j * f_ratio / (1 + f_ratio**2)
        
        # Apply to fracture-sensitive components
        C_dynamic = C_static.copy().astype(complex)
        
        # Normal compliance most sensitive to fluid flow
        C_dynamic[2, 2] *= (1 + 0.2 * dispersion_factor)
        
        # Shear components
        C_dynamic[3, 3] *= (1 + 0.05 * dispersion_factor)
        C_dynamic[4, 4] *= (1 + 0.05 * dispersion_factor)
        
        # Add attenuation
        C_dynamic *= (1 - 1j * alpha)
        
        return C_dynamic
    
    def _calculate_effective_density(self, background: Dict, 
                                    fractures: Dict) -> float:
        """Calculate effective density considering fractures and porosity"""
        rho_bg = background['rho']
        phi = background.get('porosity', 0.2)
        phi_f = fractures.get('fracture_porosity', 0.05)
        
        # Simple mixing law
        rho_eff = rho_bg * (1 - phi - phi_f)
        
        if 'fluid_density' in fractures:
            rho_eff += fractures['fluid_density'] * (phi + phi_f)
        
        return rho_eff
    
    def _calculate_anisotropy_parameters(self, C: np.ndarray, rho: float) -> Dict:
        """Calculate comprehensive anisotropy parameters"""
        
        # Extract stiffness components
        C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
        C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
        C23, C13, C12 = C[1, 2], C[0, 2], C[0, 1]
        
        # Velocities
        Vp0 = np.sqrt(C33 / rho)
        Vs0 = np.sqrt(C55 / rho)
        Vp90 = np.sqrt(C11 / rho)  # Horizontal P-wave
        
        # Thomsen-style parameters
        epsilon1 = (C11 - C33) / (2 * C33)
        epsilon2 = (C22 - C33) / (2 * C33)
        delta1 = ((C13 + C55)**2 - (C33 - C55)**2) / (2 * C33 * (C33 - C55))
        delta2 = ((C23 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44))
        gamma1 = (C66 - C44) / (2 * C44)
        gamma2 = (C66 - C55) / (2 * C55)
        
        # Fracture parameters
        ZN = (1/C33 - 1/C11)  # Normal weakness
        ZT = (1/C55 - 1/C66)  # Tangential weakness
        
        # Anisotropy strength
        P_aniso = (C11 - C33) / C33
        S_aniso = (C66 - C44) / C44
        
        return {
            'Vp0': Vp0, 'Vs0': Vs0, 'Vp90': Vp90,
            'epsilon1': epsilon1, 'epsilon2': epsilon2,
            'delta1': delta1, 'delta2': delta2,
            'gamma1': gamma1, 'gamma2': gamma2,
            'ZN': ZN, 'ZT': ZT,
            'P_anisotropy': P_aniso,
            'S_anisotropy': S_aniso,
            'C11': C11, 'C33': C33, 'C55': C55, 'C66': C66
        }
    
    def calculate_avaz_response(self,
                               theta: np.ndarray,
                               phi: np.ndarray,
                               model_params: Dict,
                               frequency: float = 30.0) -> np.ndarray:
        """
        Calculate AVAz response using orthorhombic Rüger equations
        Enhanced with frequency dependence and fluid effects
        """
        
        C = model_params['stiffness']
        rho = model_params['density']
        anisotropy = model_params['anisotropy_params']
        
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        
        # Orthorhombic Rüger approximation
        Vp0 = anisotropy['Vp0']
        Vs0 = anisotropy['Vs0']
        epsilon = anisotropy['epsilon1']
        delta = anisotropy['delta1']
        gamma = anisotropy['gamma1']
        
        # Background isotropic contrast
        Z2 = rho * Vp0
        dZ = 0.1 * Z2  # 10% contrast
        
        R_pp = np.zeros((len(theta_rad), len(phi_rad)))
        
        for i, theta_i in enumerate(theta_rad):
            sin2 = np.sin(theta_i)**2
            sin2_tan2 = sin2 * np.tan(theta_i)**2
            
            for j, phi_j in enumerate(phi_rad):
                # Isotropic term
                R0 = 0.5 * dZ / Z2
                
                # Gradient term
                G = 0.5 * (Vp0 * 0.1 / Vp0) - 2 * (Vs0/Vp0)**2 * (2 * 0.1 + 0.1)
                
                # Anisotropic terms
                B_ani = 0.5 * (delta + 2 * (2*Vs0/Vp0)**2 * gamma)
                C_ani = 0.5 * (0.1 - epsilon * np.cos(phi_j)**4 + 
                              delta * np.sin(phi_j)**2 * np.cos(phi_j)**2)
                
                # Full Rüger equation
                R_pp[i, j] = (R0 + (G + B_ani * np.cos(phi_j)**2) * sin2 + 
                             C_ani * sin2_tan2)
        
        # Apply frequency correction
        if self.frequency_dependent:
            f_corr = self._frequency_correction(theta_rad, frequency)
            R_pp *= f_corr[:, np.newaxis]
        
        return R_pp
    
    def _frequency_correction(self, theta: np.ndarray, frequency: float) -> np.ndarray:
        """Frequency-dependent correction factor"""
        f_char = self.dispersion_params['characteristic_frequency']
        ratio = frequency / f_char
        
        # Dispersion increases with angle
        correction = 1 + 0.15 * np.sin(theta) * ratio / (1 + ratio**2)
        
        return correction
    
    def calculate_elastic_impedance(self, theta: float, phi: float,
                                   model_params: Dict) -> float:
        """Calculate elastic impedance for given angle and azimuth"""
        
        C = model_params['stiffness']
        rho = model_params['density']
        
        # Simplified EI calculation
        Vp = np.sqrt(C[2, 2] / rho)
        Vs = np.sqrt(C[5, 5] / rho)
        
        # EI = Vp^a * Vs^b * ρ^c with azimuthal dependence
        K = (Vs / Vp)**2
        
        # Coefficients from Connolly
        a = np.cos(np.radians(phi))**2 + np.sin(np.radians(phi))**2 * (1 - 2 * K)
        b = -8 * K * np.sin(np.radians(phi))**2
        c = 1 - 4 * K * np.sin(np.radians(phi))**2
        
        EI = Vp**a * Vs**b * rho**c
        
        return EI


# ============================================================================
# Enhanced Synthetic Generator
# ============================================================================

class EnhancedSyntheticGenerator:
    """Generate synthetic seismic with realistic fracture effects"""
    
    def __init__(self, model: EnhancedFractureModel):
        self.model = model
        
    def generate_synthetic_data(self, config: Dict) -> Dict:
        """
        Generate comprehensive synthetic dataset
        
        Parameters:
        -----------
        config : Dict with generation parameters
        
        Returns:
        --------
        Dict with synthetic data cubes
        """
        
        # Extract dimensions
        nx, ny, nz = config['nx'], config['ny'], config['nz']
        dx, dy, dz = config['dx'], config['dy'], config['dz']
        
        # Create coordinate grids
        x = np.arange(nx) * dx
        y = np.arange(ny) * dy
        z = np.arange(nz) * dz
        
        # Generate geological model
        geology = self._generate_geological_model(nx, ny, nz, config)
        
        # Generate fracture model
        fractures = self._generate_fracture_model(nx, ny, nz, geology, config)
        
        # Generate stress field
        stress = self._generate_stress_field(nx, ny, nz, geology, config)
        
        # Calculate elastic properties
        elastic_props = self._calculate_elastic_properties(
            geology, fractures, stress, config
        )
        
        # Generate synthetic seismic
        seismic_data = self._generate_seismic_data(
            elastic_props, geology, config
        )
        
        # Generate well logs at selected locations
        well_logs = self._generate_well_logs(elastic_props, geology, config)
        
        return {
            'coordinates': {'x': x, 'y': y, 'z': z},
            'geology': geology,
            'fractures': fractures,
            'stress': stress,
            'elastic_properties': elastic_props,
            'seismic_data': seismic_data,
            'well_logs': well_logs,
            'config': config
        }
    
    def _generate_geological_model(self, nx: int, ny: int, nz: int,
                                  config: Dict) -> Dict:
        """Generate realistic geological model with layers and faults"""
        
        # Create depth trend
        z_coords = np.arange(nz) * config['dz']
        
        # Base velocity trend
        Vp_base = 2000 + 1.5 * z_coords  # Increase with depth
        
        # Add layers
        n_layers = config.get('n_layers', 5)
        layer_thickness = nz // n_layers
        
        Vp = np.ones((nz, ny, nx)) * Vp_base[:, np.newaxis, np.newaxis]
        Vs = Vp / 1.8  # Simple Vp/Vs ratio
        rho = 2000 + 0.3 * Vp  # Gardner's relation
        
        # Add layer variations
        for i in range(n_layers):
            layer_start = i * layer_thickness
            layer_end = min((i + 1) * layer_thickness, nz)
            
            # Random layer properties
            layer_Vp_shift = np.random.uniform(-200, 200)
            layer_Vs_shift = layer_Vp_shift / 1.8
            layer_rho_shift = 0.3 * layer_Vp_shift
            
            Vp[layer_start:layer_end, :, :] += layer_Vp_shift
            Vs[layer_start:layer_end, :, :] += layer_Vs_shift
            rho[layer_start:layer_end, :, :] += layer_rho_shift
        
        # Add faults
        if config.get('add_faults', True):
            fault_dip = np.radians(config.get('fault_dip', 60))
            fault_strike = np.radians(config.get('fault_strike', 30))
            
            for fault_pos in [nx//4, nx//2, 3*nx//4]:
                # Create fault plane
                for i in range(nz):
                    offset = int(i * np.tan(fault_dip))
                    if 0 <= fault_pos + offset < nx:
                        # Offset velocities across fault
                        Vp[i, :, fault_pos+offset:] *= 0.95  # 5% reduction
                        Vs[i, :, fault_pos+offset:] *= 0.95
                        rho[i, :, fault_pos+offset:] *= 0.98
        
        # Add channels or other features
        if config.get('add_channels', True):
            # Create sinuous channel
            channel_center = ny // 2 + 20 * np.sin(2 * np.pi * np.arange(nx) / nx)
            
            for i in range(nx):
                channel_width = 5
                channel_y = int(channel_center[i])
                y_start = max(0, channel_y - channel_width)
                y_end = min(ny, channel_y + channel_width)
                
                # Channel properties (sand)
                Vp[nz//3:nz//2, y_start:y_end, i] *= 0.9  # Slower in sand
                rho[nz//3:nz//2, y_start:y_end, i] *= 0.95
        
        return {
            'Vp': Vp,
            'Vs': Vs,
            'rho': rho,
            'porosity': np.clip(0.4 - 0.0003 * Vp, 0.05, 0.35),
            'clay_content': np.random.uniform(0.1, 0.5, (nz, ny, nx))
        }
    
    def _generate_fracture_model(self, nx: int, ny: int, nz: int,
                                geology: Dict, config: Dict) -> Dict:
        """Generate realistic fracture model"""
        
        np.random.seed(config.get('random_seed', 42))
        
        # Base fracture density correlated with geology
        fracture_density = np.zeros((nz, ny, nx))
        
        # Higher fractures in brittle layers (low clay, high Vp/Vs)
        brittleness = (1 - geology['clay_content']) * (geology['Vp'] / geology['Vs'])
        brittleness = (brittleness - brittleness.min()) / (brittleness.max() - brittleness.min())
        
        # Fractures follow brittleness and structure
        from scipy.ndimage import gaussian_filter
        
        # Create correlated random field
        noise = np.random.randn(nz, ny, nx)
        correlated_field = gaussian_filter(noise, sigma=config.get('fracture_correlation', 3.0))
        
        # Combine with brittleness
        fracture_density = 0.05 + 0.2 * (0.7 * brittleness + 0.3 * correlated_field)
        fracture_density = np.clip(fracture_density, 0.01, 0.3)
        
        # Fracture orientation field
        # Regional trend with local variations
        orientation = np.zeros((nz, ny, nx))
        
        regional_trend = np.radians(config.get('regional_trend', 45))
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    # Base orientation with regional trend
                    base_orientation = regional_trend
                    
                    # Add local structural variations
                    local_variation = 0.3 * np.sin(2 * np.pi * k / 50) * np.cos(2 * np.pi * j / 40)
                    
                    # Add fault-related rotation near faults
                    fault_distance = np.sqrt((k - nx/3)**2 + (j - ny/2)**2)
                    fault_effect = 0.5 * np.exp(-fault_distance / 30)
                    
                    orientation[i, j, k] = base_orientation + local_variation + fault_effect
        
        # Fracture aspect ratio (typically 0.001-0.1)
        aspect_ratio = 0.005 + 0.015 * np.random.rand(nz, ny, nx)
        
        # Fracture fill (fluid/gas)
        fracture_fill = np.full((nz, ny, nx), 'fluid', dtype=object)
        
        # Gas in anticlines or high structures
        anticline_center = nx // 2
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    distance = np.sqrt((k - anticline_center)**2 + (j - ny/2)**2)
                    if distance < 20 and i > nz//2:
                        fracture_fill[i, j, k] = 'gas' if np.random.random() > 0.7 else 'fluid'
        
        return {
            'density': fracture_density,
            'orientation': orientation,
            'aspect_ratio': aspect_ratio,
            'fill': fracture_fill,
            'dip': np.full((nz, ny, nx), np.radians(90))  # Vertical fractures
        }
    
    def _generate_stress_field(self, nx: int, ny: int, nz: int,
                              geology: Dict, config: Dict) -> Dict:
        """Generate realistic stress field"""
        
        z_coords = np.arange(nz) * config['dz']
        
        # Create base 1D stress profile (varies with depth only)
        sigma_v_base = -0.022 * z_coords  # MPa/m gradient
        
        # Convert to 3D arrays using broadcasting
        sigma_v = sigma_v_base[:, np.newaxis, np.newaxis]  # Shape: (nz, 1, 1)
        
        # Horizontal stresses
        sigma_h = sigma_v * config.get('k0_min', 0.7)  # Minimum horizontal
        sigma_H = sigma_v * config.get('k0_max', 0.9)  # Maximum horizontal
        
        # Broadcast to full 3D arrays
        sigma_v_3d = np.broadcast_to(sigma_v, (nz, ny, nx))
        sigma_h_3d = np.broadcast_to(sigma_h, (nz, ny, nx))
        sigma_H_3d = np.broadcast_to(sigma_H, (nz, ny, nx)).copy()  # Need copy for modification
        
        # Add spatial variations - stress concentration near faults
        for fault_pos in [nx//4, nx//2, 3*nx//4]:
            for i in range(nx):
                distance = abs(i - fault_pos)
                if distance < 10:
                    stress_concentration = np.exp(-distance/5)
                    sigma_H_3d[:, :, i] *= (1 + 0.2 * stress_concentration)
        
        # Stress orientation (azimuth of sigma_H)
        stress_orientation = np.full((nz, ny, nx), np.radians(45))  # N45E
        
        return {
            'vertical': sigma_v_3d,
            'horizontal_min': sigma_h_3d,
            'horizontal_max': sigma_H_3d,
            'orientation': stress_orientation,
            'pore_pressure': sigma_v_3d * 0.4  # Hydrostatic
        }
    
    def _calculate_elastic_properties(self, geology: Dict, fractures: Dict,
                                     stress: Dict, config: Dict) -> Dict:
        """Calculate elastic properties for each grid cell"""
        
        nz, ny, nx = geology['Vp'].shape
        
        # Initialize arrays for stiffness tensors
        C = np.zeros((nz, ny, nx, 6, 6))
        density = np.zeros((nz, ny, nx))
        anisotropy = {
            'epsilon1': np.zeros((nz, ny, nx)),
            'gamma1': np.zeros((nz, ny, nx)),
            'ZN': np.zeros((nz, ny, nx)),
            'ZT': np.zeros((nz, ny, nx))
        }
        
        # Calculate for each cell (can be parallelized)
        for i in range(nz):
            for j in range(ny):
                for k in range(nx):
                    # Background rock
                    background = {
                        'Vp': geology['Vp'][i, j, k],
                        'Vs': geology['Vs'][i, j, k],
                        'rho': geology['rho'][i, j, k],
                        'porosity': geology['porosity'][i, j, k],
                        'clay': geology['clay_content'][i, j, k]
                    }
                    
                    # Fracture properties
                    frac = {
                        'density': fractures['density'][i, j, k],
                        'aspect_ratio': fractures['aspect_ratio'][i, j, k],
                        'orientation': fractures['orientation'][i, j, k],
                        'dip': fractures['dip'][i, j, k],
                        'fill': fractures['fill'][i, j, k]
                    }
                    
                    # Stress at this location
                    stress_local = {
                        'vertical': stress['vertical'][i, j, k],
                        'horizontal': stress['horizontal_min'][i, j, k],
                        'horizontal_max': stress['horizontal_max'][i, j, k]
                    }
                    
                    # Fluid properties for substitution
                    fill_type = frac['fill']
                    if fill_type in self.model.fluid_properties:
                        fluid = self.model.fluid_properties[fill_type]
                    else:
                        fluid = self.model.fluid_properties['brine']  # Default
                    
                    fluid_props = {
                        'porosity': background['porosity'],
                        'fluid_K': fluid['K'],
                        'fluid_density': fluid['rho'],
                        'mineral_K': 37e9,  # Quartz
                        'mineral_G': 44e9
                    }
                    
                    # Calculate effective properties
                    result = self.model.effective_medium_model(
                        background, frac, stress_local, fluid_props
                    )
                    
                    C[i, j, k] = result['stiffness']
                    density[i, j, k] = result['density']
                    anisotropy['epsilon1'][i, j, k] = result['anisotropy_params']['epsilon1']
                    anisotropy['gamma1'][i, j, k] = result['anisotropy_params']['gamma1']
                    anisotropy['ZN'][i, j, k] = result['anisotropy_params']['ZN']
                    anisotropy['ZT'][i, j, k] = result['anisotropy_params']['ZT']
        
        return {
            'stiffness': C,
            'density': density,
            'anisotropy': anisotropy
        }
    
    def _generate_seismic_data(self, elastic_props: Dict, geology: Dict,
                              config: Dict) -> Dict:
        """Generate synthetic seismic data for multiple azimuths"""
        
        nz, ny, nx = geology['Vp'].shape
        
        # Define acquisition geometry
        azimuths = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330])
        incidence_angles = np.array([0, 10, 20, 30, 40])
        
        # Generate wavelet
        wavelet_freq = config.get('wavelet_freq', 30)
        wavelet = self._ricker_wavelet(wavelet_freq, dt=config.get('dt', 0.001))
        
        # Initialize seismic cubes
        seismic_cubes = {}
        
        for az in azimuths:
            # Calculate reflectivity for this azimuth
            R = self._calculate_reflectivity_cube(
                elastic_props, geology, incidence_angles, az
            )
            
            # Convolve with wavelet
            seismic = self._convolve_with_wavelet(R, wavelet)
            
            seismic_cubes[f'azimuth_{az}'] = seismic
        
        return {
            'cubes': seismic_cubes,
            'azimuths': azimuths,
            'incidence_angles': incidence_angles,
            'wavelet': wavelet
        }
    
    def _calculate_reflectivity_cube(self, elastic_props: Dict, geology: Dict,
                                    incidence_angles: np.ndarray,
                                    azimuth: float) -> np.ndarray:
        """Calculate reflectivity cube for given azimuth"""
        
        nz, ny, nx = geology['Vp'].shape
        n_angles = len(incidence_angles)
        
        R = np.zeros((n_angles, nz, ny, nx))
        
        # Simplified reflectivity calculation
        # In practice, would use full anisotropic Zoeppritz or Rüger
        C = elastic_props['stiffness']
        rho = elastic_props['density']
        
        for angle_idx, theta in enumerate(incidence_angles):
            theta_rad = np.radians(theta)
            az_rad = np.radians(azimuth)
            
            for i in range(1, nz):
                # Calculate impedances
                Z1 = rho[i-1] * np.sqrt(C[i-1, ..., 2, 2] / rho[i-1])
                Z2 = rho[i] * np.sqrt(C[i, ..., 2, 2] / rho[i])
                
                # Isotropic reflectivity
                R0 = (Z2 - Z1) / (Z2 + Z1)
                
                # Azimuthal variation
                # Use anisotropy parameters
                epsilon1 = elastic_props['anisotropy']['epsilon1'][i]
                gamma1 = elastic_props['anisotropy']['gamma1'][i]
                
                # Simplified azimuthal term
                az_term = (epsilon1 * np.cos(az_rad)**2 + 
                          gamma1 * np.sin(az_rad)**2) * np.sin(theta_rad)**2
                
                R[angle_idx, i] = R0 + 0.1 * az_term
        
        return R
    
    def _ricker_wavelet(self, frequency: float, dt: float = 0.001) -> np.ndarray:
        """Generate Ricker wavelet"""
        t = np.arange(-0.1, 0.1, dt)
        wavelet = (1 - 2 * (np.pi * frequency * t)**2) * \
                 np.exp(-(np.pi * frequency * t)**2)
        return wavelet / np.max(np.abs(wavelet))
    
    def _convolve_with_wavelet(self, R: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
        """Convolve reflectivity with wavelet"""
        n_angles, nz, ny, nx = R.shape
        n_wavelet = len(wavelet)
        
        seismic = np.zeros((n_angles, nz + n_wavelet - 1, ny, nx))
        
        for angle_idx in range(n_angles):
            for j in range(ny):
                for k in range(nx):
                    trace = R[angle_idx, :, j, k]
                    seismic[angle_idx, :, j, k] = np.convolve(trace, wavelet, mode='full')
        
        # Trim to original length
        trim_start = n_wavelet // 2
        trim_end = -(n_wavelet // 2 - 1)
        
        return seismic[:, trim_start:trim_end, :, :]
    
    def _generate_well_logs(self, elastic_props: Dict, geology: Dict,
                           config: Dict) -> Dict:
        """Generate synthetic well logs at selected locations"""
        
        nz, ny, nx = geology['Vp'].shape
        
        # Well locations
        well_locations = [
            {'x': nx//4, 'y': ny//2, 'name': 'Well_A'},
            {'x': nx//2, 'y': ny//2, 'name': 'Well_B'},
            {'x': 3*nx//4, 'y': ny//2, 'name': 'Well_C'}
        ]
        
        well_logs = {}
        
        for well in well_locations:
            x, y = well['x'], well['y']
            
            logs = {
                'depth': np.arange(nz) * config['dz'],
                'Vp': geology['Vp'][:, y, x],
                'Vs': geology['Vs'][:, y, x],
                'rho': geology['rho'][:, y, x],
                'porosity': geology['porosity'][:, y, x],
                'clay': geology['clay_content'][:, y, x],
                'fracture_density': elastic_props['anisotropy']['ZN'][:, y, x],
                'anisotropy_strength': elastic_props['anisotropy']['epsilon1'][:, y, x]
            }
            
            # Calculate derived properties
            logs['Vp/Vs'] = logs['Vp'] / logs['Vs']
            logs['Acoustic_Impedance'] = logs['Vp'] * logs['rho']
            logs['Shear_Impedance'] = logs['Vs'] * logs['rho']
            logs['LambdaRho'] = logs['rho'] * (logs['Vp']**2 - 2 * logs['Vs']**2)
            logs['MuRho'] = logs['rho'] * logs['Vs']**2
            
            well_logs[well['name']] = logs
        
        return well_logs


# ============================================================================
# Enhanced Visualization and Analysis
# ============================================================================

class EnhancedVisualization:
    """Enhanced visualization tools for fracture analysis"""
    
    @staticmethod
    def plot_comprehensive_results(results: Dict):
        """Create comprehensive visualization of results"""
        
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot layout
        gs = plt.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Fracture density map
        ax1 = fig.add_subplot(gs[0, 0])
        z_slice = results['fractures']['density'].shape[0] // 2
        im1 = ax1.imshow(results['fractures']['density'][z_slice], 
                        cmap='viridis', aspect='auto')
        ax1.set_title('Fracture Density - Map View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1)
        
        # 2. Fracture orientation rose diagram
        ax2 = fig.add_subplot(gs[0, 1], projection='polar')
        orientations = results['fractures']['orientation'][z_slice].flatten()
        hist, bins = np.histogram(orientations, bins=36, range=(0, 2*np.pi))
        centers = 0.5 * (bins[:-1] + bins[1:])
        width = 2 * np.pi / 36
        ax2.bar(centers, hist, width=width, alpha=0.7, color='steelblue')
        ax2.set_title('Fracture Orientation', pad=20)
        
        # 3. Anisotropy strength
        ax3 = fig.add_subplot(gs[0, 2])
        ani_strength = results['elastic_properties']['anisotropy']['epsilon1'][z_slice]
        im3 = ax3.imshow(ani_strength, cmap='seismic', aspect='auto', 
                        vmin=-0.2, vmax=0.2)
        ax3.set_title('P-Wave Anisotropy (ε)')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        plt.colorbar(im3, ax=ax3)
        
        # 4. Cross-section
        ax4 = fig.add_subplot(gs[0, 3])
        y_slice = results['geology']['Vp'].shape[1] // 2
        vp_section = results['geology']['Vp'][:, y_slice, :]
        im4 = ax4.imshow(vp_section.T, aspect='auto', cmap='jet',
                        extent=[0, vp_section.shape[0], 0, vp_section.shape[1]])
        ax4.set_title('Vp Cross-section')
        ax4.set_xlabel('Depth')
        ax4.set_ylabel('X')
        plt.colorbar(im4, ax=ax4)
        
        # 5-7. Seismic amplitude for different azimuths
        seismic_cubes = results['seismic_data']['cubes']
        azimuths = results['seismic_data']['azimuths'][:3]  # First 3 azimuths
        
        for i, az in enumerate(azimuths):
            ax = fig.add_subplot(gs[1, i])
            seismic = seismic_cubes[f'azimuth_{az}'][2, z_slice]  # 20° incidence
            vmax = np.percentile(np.abs(seismic), 95)
            im = ax.imshow(seismic, cmap='gray', aspect='auto',
                          vmin=-vmax, vmax=vmax)
            ax.set_title(f'Seismic - Azimuth {az}°')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
        
        # 8. AVAz response at a point
        ax8 = fig.add_subplot(gs[1, 3])
        center_x = results['geology']['Vp'].shape[2] // 2
        center_y = results['geology']['Vp'].shape[1] // 2
        
        # Extract amplitudes for rose plot
        amplitudes = []
        for az in results['seismic_data']['azimuths']:
            seismic = seismic_cubes[f'azimuth_{az}'][2, z_slice, 
                                                    center_y-5:center_y+5,
                                                    center_x-5:center_x+5]
            amplitudes.append(np.mean(np.abs(seismic)))
        
        amplitudes = np.array(amplitudes)
        amplitudes = amplitudes / np.max(amplitudes)
        
        ax8.plot(results['seismic_data']['azimuths'], amplitudes, 'ro-', linewidth=2)
        ax8.fill_between(results['seismic_data']['azimuths'], 0, amplitudes, alpha=0.3)
        ax8.set_xlabel('Azimuth (degrees)')
        ax8.set_ylabel('Normalized Amplitude')
        ax8.set_title('Azimuthal Amplitude Variation')
        ax8.grid(True, alpha=0.3)
        
        # 9-11. Well logs
        well_logs = results['well_logs']
        well_names = list(well_logs.keys())
        
        for i, well_name in enumerate(well_names[:3]):
            ax = fig.add_subplot(gs[2, i])
            logs = well_logs[well_name]
            
            ax.plot(logs['Vp']/1000, logs['depth'], 'b-', label='Vp')
            ax.plot(logs['Vs']/1000, logs['depth'], 'r-', label='Vs')
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Depth')
            ax.set_title(f'{well_name} - Velocity')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # 12. Stress profile
        ax12 = fig.add_subplot(gs[2, 3])
        stress = results['stress']
        depth = np.arange(stress['vertical'].shape[0]) * results['config']['dz']
        center_y = stress['vertical'].shape[1] // 2
        center_x = stress['vertical'].shape[2] // 2
        
        ax12.plot(-stress['vertical'][:, center_y, center_x], depth, 'k-', label='σv', linewidth=2)
        ax12.plot(-stress['horizontal_max'][:, center_y, center_x], depth, 'r-', label='σH', linewidth=2)
        ax12.plot(-stress['horizontal_min'][:, center_y, center_x], depth, 'b-', label='σh', linewidth=2)
        ax12.plot(-stress['pore_pressure'][:, center_y, center_x], depth, 'g-', label='Pp', linewidth=2)
        
        ax12.set_xlabel('Stress (MPa)')
        ax12.set_ylabel('Depth (m)')
        ax12.set_title('Stress Profile')
        ax12.invert_yaxis()
        ax12.legend()
        ax12.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Fracture Analysis Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_interactive_3d_plot(results: Dict):
        """Create interactive 3D visualization using plotly"""
        
        z_slice = results['fractures']['density'].shape[0] // 2
        fracture_density = results['fractures']['density'][z_slice]
        orientation = results['fractures']['orientation'][z_slice]
        
        # Create meshgrid
        ny, nx = fracture_density.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        
        # Create figure with multiple surfaces
        fig = go.Figure()
        
        # Fracture density surface
        fig.add_trace(go.Surface(
            z=fracture_density,
            x=X,
            y=Y,
            colorscale='Viridis',
            name='Fracture Density',
            opacity=0.9,
            contours_z=dict(show=True, project_z=True)
        ))
        
        # Orientation vectors
        # Create quiver plot for orientation
        scale = 5
        skip = 4  # Plot every 4th vector
        
        U = np.cos(orientation[::skip, ::skip]) * scale
        V = np.sin(orientation[::skip, ::skip]) * scale
        X_vec = X[::skip, ::skip].flatten()
        Y_vec = Y[::skip, ::skip].flatten()
        Z_vec = fracture_density[::skip, ::skip].flatten()
        U = U.flatten()
        V = V.flatten()
        
        # Add cone vectors for orientation
        fig.add_trace(go.Cone(
            x=X_vec,
            y=Y_vec,
            z=Z_vec,
            u=U,
            v=V,
            w=np.zeros_like(Z_vec),
            sizemode="absolute",
            sizeref=2,
            showscale=False,
            name='Fracture Orientation'
        ))
        
        fig.update_layout(
            title='3D Fracture Model',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Fracture Density',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.8)
                )
            ),
            height=700
        )
        
        return fig


# ============================================================================
# Streamlit Application
# ============================================================================

def create_streamlit_app():
    """Create comprehensive Streamlit application for AVAz analysis"""
    
    st.set_page_config(layout="wide", page_title="Enhanced AVAz Analysis")
    st.title("Enhanced AVAz Analysis with Fracture Characterization")
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = EnhancedFractureModel(
            model_type='orthorhombic',
            use_geomechanics=True,
            frequency_dependent=True,
            use_fluid_substitution=True
        )
    
    if 'generator' not in st.session_state:
        st.session_state.generator = EnhancedSyntheticGenerator(st.session_state.model)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Model Configuration", "Synthetic Generation", 
         "AVAz Analysis", "Fluid Substitution", "Inversion"]
    )
    
    if app_mode == "Model Configuration":
        st.header("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Fracture Model")
            model_type = st.selectbox(
                "Model Type",
                ['orthorhombic', 'HTI', 'VTI_HTI', 'monoclinic'],
                index=0
            )
            
            use_geomechanics = st.checkbox("Include Stress Sensitivity", True)
            frequency_dependent = st.checkbox("Include Frequency Effects", True)
            use_fluid_sub = st.checkbox("Enable Fluid Substitution", True)
            
            # Update model
            st.session_state.model = EnhancedFractureModel(
                model_type=model_type,
                use_geomechanics=use_geomechanics,
                frequency_dependent=frequency_dependent,
                use_fluid_substitution=use_fluid_sub
            )
            st.success("Model updated successfully!")
        
        with col2:
            st.subheader("Physical Properties")
            
            st.markdown("**Fluid Properties**")
            fluid_type = st.selectbox("Fluid Type", ['brine', 'oil', 'gas'], index=0)
            fluid = st.session_state.model.fluid_properties[fluid_type]
            
            st.write(f"Density: {fluid['rho']} kg/m³")
            st.write(f"Bulk Modulus: {fluid['K']/1e9:.1f} GPa")
            st.write(f"Viscosity: {fluid['mu']} Pa·s")
    
    elif app_mode == "Synthetic Generation":
        st.header("Synthetic Data Generation")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generation Parameters")
            
            nx = st.slider("Grid X Size", 50, 200, 100)
            ny = st.slider("Grid Y Size", 50, 200, 100)
            nz = st.slider("Grid Z Size", 30, 100, 50)
            
            dx = st.slider("X Spacing (m)", 10.0, 50.0, 25.0)
            dy = st.slider("Y Spacing (m)", 10.0, 50.0, 25.0)
            dz = st.slider("Z Spacing (m)", 2.0, 10.0, 5.0)
            
            add_faults = st.checkbox("Add Faults", True)
            add_channels = st.checkbox("Add Channels", True)
            
            wavelet_freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 30)
            
            if st.button("Generate Synthetic Data", type="primary"):
                with st.spinner("Generating synthetic data..."):
                    config = {
                        'nx': nx, 'ny': ny, 'nz': nz,
                        'dx': dx, 'dy': dy, 'dz': dz,
                        'add_faults': add_faults,
                        'add_channels': add_channels,
                        'wavelet_freq': wavelet_freq,
                        'n_layers': 5
                    }
                    
                    st.session_state.synthetic_results = (
                        st.session_state.generator.generate_synthetic_data(config)
                    )
                    st.success("Synthetic data generated successfully!")
        
        with col2:
            if 'synthetic_results' in st.session_state:
                st.subheader("Results Visualization")
                
                # Quick view of key results
                results = st.session_state.synthetic_results
                
                tab1, tab2, tab3 = st.tabs(["Map View", "Cross-section", "Well Logs"])
                
                with tab1:
                    z_slice = results['fractures']['density'].shape[0] // 2
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Fracture density
                    im1 = ax1.imshow(results['fractures']['density'][z_slice], 
                                    cmap='viridis', aspect='auto')
                    ax1.set_title('Fracture Density')
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Vp map
                    im2 = ax2.imshow(results['geology']['Vp'][z_slice], 
                                    cmap='jet', aspect='auto')
                    ax2.set_title('P-Wave Velocity')
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    plt.colorbar(im2, ax=ax2)
                    
                    st.pyplot(fig)
                
                with tab2:
                    y_slice = results['geology']['Vp'].shape[1] // 2
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Vp cross-section
                    vp_section = results['geology']['Vp'][:, y_slice, :]
                    im1 = ax1.imshow(vp_section.T, aspect='auto', cmap='jet')
                    ax1.set_title('Vp Cross-section')
                    ax1.set_xlabel('Depth')
                    ax1.set_ylabel('X')
                    plt.colorbar(im1, ax=ax1)
                    
                    # Fracture density cross-section
                    frac_section = results['fractures']['density'][:, y_slice, :]
                    im2 = ax2.imshow(frac_section.T, aspect='auto', cmap='viridis')
                    ax2.set_title('Fracture Density Cross-section')
                    ax2.set_xlabel('Depth')
                    ax2.set_ylabel('X')
                    plt.colorbar(im2, ax=ax2)
                    
                    st.pyplot(fig)
                
                with tab3:
                    well_name = st.selectbox("Select Well", 
                                           list(results['well_logs'].keys()))
                    
                    if well_name:
                        logs = results['well_logs'][well_name]
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
                        
                        # Vp and Vs
                        ax1 = axes[0]
                        ax1.plot(logs['Vp']/1000, logs['depth'], 'b-', label='Vp')
                        ax1.plot(logs['Vs']/1000, logs['depth'], 'r-', label='Vs')
                        ax1.set_xlabel('Velocity (km/s)')
                        ax1.set_ylabel('Depth (m)')
                        ax1.set_title(f'{well_name} - Velocities')
                        ax1.invert_yaxis()
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        
                        # Impedances
                        ax2 = axes[1]
                        ax2.plot(logs['Acoustic_Impedance']/1e6, logs['depth'], 'b-', 
                                label='AI')
                        ax2.plot(logs['Shear_Impedance']/1e6, logs['depth'], 'r-', 
                                label='SI')
                        ax2.set_xlabel('Impedance (MPa·s/m)')
                        ax2.set_ylabel('Depth (m)')
                        ax2.set_title(f'{well_name} - Impedances')
                        ax2.invert_yaxis()
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        
                        # Fracture and anisotropy
                        ax3 = axes[2]
                        ax3.plot(logs['fracture_density'], logs['depth'], 'g-', 
                                label='Fracture Density')
                        ax3_twin = ax3.twinx()
                        ax3_twin.plot(logs['anisotropy_strength'], logs['depth'], 'm-',
                                     label='Anisotropy')
                        ax3.set_xlabel('Fracture Density')
                        ax3_twin.set_xlabel('Anisotropy (ε)')
                        ax3.set_ylabel('Depth (m)')
                        ax3.set_title(f'{well_name} - Fracture Properties')
                        ax3.invert_yaxis()
                        ax3.grid(True, alpha=0.3)
                        
                        # Combine legends
                        lines1, labels1 = ax3.get_legend_handles_labels()
                        lines2, labels2 = ax3_twin.get_legend_handles_labels()
                        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        
                        st.pyplot(fig)
    
    elif app_mode == "AVAz Analysis":
        st.header("AVAz Analysis")
        
        if 'synthetic_results' not in st.session_state:
            st.warning("Please generate synthetic data first!")
            st.info("Go to 'Synthetic Generation' mode to create a synthetic dataset.")
            return
        
        results = st.session_state.synthetic_results
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Analysis Parameters")
            
            # Select location
            nx, ny = results['geology']['Vp'].shape[2], results['geology']['Vp'].shape[1]
            x_pos = st.slider("X Position", 0, nx-1, nx//2)
            y_pos = st.slider("Y Position", 0, ny-1, ny//2)
            z_pos = st.slider("Z Position (depth slice)", 0, 
                             results['geology']['Vp'].shape[0]-1,
                             results['geology']['Vp'].shape[0]//2)
            
            # Select angles and azimuths
            max_angle = st.slider("Maximum Incidence Angle", 10, 50, 40)
            angles = np.linspace(0, max_angle, 10)
            azimuth_step = st.slider("Azimuth Step", 5, 30, 15)
            azimuths = np.arange(0, 360, azimuth_step)
            
            if st.button("Calculate AVAz Response", type="primary"):
                with st.spinner("Calculating AVAz..."):
                    # Extract properties at selected location
                    background = {
                        'Vp': results['geology']['Vp'][z_pos, y_pos, x_pos],
                        'Vs': results['geology']['Vs'][z_pos, y_pos, x_pos],
                        'rho': results['geology']['rho'][z_pos, y_pos, x_pos],
                        'porosity': results['geology']['porosity'][z_pos, y_pos, x_pos]
                    }
                    
                    fractures = {
                        'density': results['fractures']['density'][z_pos, y_pos, x_pos],
                        'orientation': results['fractures']['orientation'][z_pos, y_pos, x_pos],
                        'aspect_ratio': results['fractures']['aspect_ratio'][z_pos, y_pos, x_pos],
                        'fill': results['fractures']['fill'][z_pos, y_pos, x_pos]
                    }
                    
                    stress = {
                        'vertical': results['stress']['vertical'][z_pos, y_pos, x_pos],
                        'horizontal': results['stress']['horizontal_min'][z_pos, y_pos, x_pos],
                        'horizontal_max': results['stress']['horizontal_max'][z_pos, y_pos, x_pos]
                    }
                    
                    # Calculate effective properties
                    model_result = st.session_state.model.effective_medium_model(
                        background, fractures, stress
                    )
                    
                    # Calculate AVAz response
                    R_pp = st.session_state.model.calculate_avaz_response(
                        angles, azimuths, model_result
                    )
                    
                    st.session_state.avaz_result = {
                        'R_pp': R_pp,
                        'angles': angles,
                        'azimuths': azimuths,
                        'location': (x_pos, y_pos, z_pos),
                        'model_result': model_result
                    }
                    st.success("AVAz calculated successfully!")
        
        with col2:
            if 'avaz_result' in st.session_state:
                st.subheader("AVAz Results")
                
                result = st.session_state.avaz_result
                R_pp = result['R_pp']
                
                tab1, tab2, tab3 = st.tabs(["3D Surface", "2D Plots", "Attributes"])
                
                with tab1:
                    # 3D surface plot
                    fig = go.Figure(data=[go.Surface(
                        z=R_pp,
                        x=result['azimuths'],
                        y=result['angles'],
                        colorscale='jet',
                        contours={
                            "z": {"show": True, "usecolormap": True, 
                                 "project": {"z": True}}
                        }
                    )])
                    
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='Azimuth (deg)',
                            yaxis_title='Incidence Angle (deg)',
                            zaxis_title='Reflectivity',
                            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                        ),
                        title=f"AVAz Response at Location ({result['location'][0]}, {result['location'][1]}, {result['location'][2]})",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # 2D plots
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Cartesian plot for selected angles
                    selected_angles = [0, 10, 20, 30, 40]
                    angle_indices = [np.argmin(np.abs(result['angles'] - ang)) 
                                    for ang in selected_angles if ang <= max(result['angles'])]
                    
                    ax1 = axes[0]
                    for idx in angle_indices:
                        angle = result['angles'][idx]
                        ax1.plot(result['azimuths'], R_pp[idx, :], 
                                label=f'{angle:.0f}°', linewidth=2)
                    
                    ax1.set_xlabel('Azimuth (degrees)')
                    ax1.set_ylabel('Reflection Coefficient')
                    ax1.set_title('AVAz for Different Incidence Angles')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Polar plot for selected azimuths
                    ax2 = plt.subplot(122, projection='polar')
                    
                    selected_azimuths = [0, 45, 90, 135, 180]
                    az_indices = [np.argmin(np.abs(result['azimuths'] - az)) 
                                 for az in selected_azimuths]
                    
                    for idx in az_indices:
                        azimuth = result['azimuths'][idx]
                        ax2.plot(np.radians(result['angles']), R_pp[:, idx], 
                                label=f'{azimuth:.0f}°', linewidth=2)
                    
                    ax2.set_title('AVAz for Different Azimuths', pad=20)
                    ax2.legend(bbox_to_anchor=(1.1, 1.0))
                    
                    st.pyplot(fig)
                
                with tab3:
                    # Extract AVAz attributes
                    st.subheader("AVAz Attributes")
                    
                    # Fit sinusoidal variation: R(φ) = A + B*cos(2*(φ - φ₀))
                    phi_rad = np.radians(result['azimuths'])
                    
                    attributes = []
                    for i, angle in enumerate(result['angles']):
                        R_angle = R_pp[i, :]
                        
                        # Fourier decomposition
                        coeffs = np.fft.fft(R_angle)
                        
                        A = np.real(coeffs[0]) / len(R_angle)  # Isotropic component
                        B = 2 * np.abs(coeffs[2]) / len(R_angle)  # 2φ component
                        phi0 = 0.5 * np.angle(coeffs[2])  # Fast direction
                        
                        attributes.append({
                            'angle': angle,
                            'A': A,
                            'B': B,
                            'phi0': np.degrees(phi0),
                            'B/A': B/A if A != 0 else 0
                        })
                    
                    # Create dataframe
                    df_attributes = pd.DataFrame(attributes)
                    st.dataframe(df_attributes.style.format({
                        'A': '{:.4f}',
                        'B': '{:.4f}',
                        'phi0': '{:.1f}',
                        'B/A': '{:.3%}'
                    }))
                    
                    # Plot attributes vs angle
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    ax1 = axes[0]
                    ax1.plot(df_attributes['angle'], df_attributes['B'], 'bo-', 
                            label='Anisotropic Magnitude (B)')
                    ax1.set_xlabel('Incidence Angle (deg)')
                    ax1.set_ylabel('B')
                    ax1.set_title('Anisotropy Magnitude vs Angle')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
                    
                    ax2 = axes[1]
                    ax2.plot(df_attributes['angle'], df_attributes['B/A'], 'ro-',
                            label='Relative Anisotropy (B/A)')
                    ax2.set_xlabel('Incidence Angle (deg)')
                    ax2.set_ylabel('B/A')
                    ax2.set_title('Relative Anisotropy vs Angle')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()
                    
                    st.pyplot(fig)
    
    elif app_mode == "Fluid Substitution":
        st.header("Brown-Korringa Fluid Substitution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Parameters")
            
            # Background rock
            st.markdown("**Background Rock Properties**")
            Vp = st.number_input("Vp (m/s)", value=3000.0)
            Vs = st.number_input("Vs (m/s)", value=1500.0)
            rho = st.number_input("Density (kg/m³)", value=2400.0)
            porosity = st.slider("Porosity", 0.01, 0.4, 0.2, 0.01)
            
            # Anisotropy parameters
            st.markdown("**Anisotropy Parameters**")
            epsilon = st.number_input("ε (Epsilon)", value=0.1, step=0.01)
            delta = st.number_input("δ (Delta)", value=0.05, step=0.01)
            gamma = st.number_input("γ (Gamma)", value=0.08, step=0.01)
            
            # Mineral properties
            st.markdown("**Mineral Properties**")
            K_min = st.number_input("Mineral K (GPa)", value=37.0, step=1.0)
            G_min = st.number_input("Mineral G (GPa)", value=44.0, step=1.0)
            
            # Fluid properties
            st.markdown("**Fluid Properties**")
            fluid_type = st.selectbox("Fluid Type", ['brine', 'oil', 'gas'])
            fluid = st.session_state.model.fluid_properties[fluid_type]
            K_fluid = fluid['K'] / 1e9  # Convert to GPa
            rho_fluid = fluid['rho']
            
            st.write(f"Fluid K: {K_fluid:.1f} GPa")
            st.write(f"Fluid Density: {rho_fluid} kg/m³")
        
        with col2:
            if st.button("Run Fluid Substitution", type="primary"):
                with st.spinner("Calculating fluid substitution..."):
                    # Create background model
                    background = {
                        'Vp': Vp,
                        'Vs': Vs,
                        'rho': rho,
                        'epsilon': epsilon,
                        'delta': delta,
                        'gamma': gamma,
                        'porosity': porosity
                    }
                    
                    # Fluid properties
                    fluid_props = {
                        'porosity': porosity,
                        'mineral_K': K_min * 1e9,
                        'mineral_G': G_min * 1e9,
                        'fluid_K': K_fluid * 1e9,
                        'fluid_density': rho_fluid
                    }
                    
                    # Calculate initial stiffness
                    C_initial = st.session_state.model._background_stiffness(background)
                    C_initial = st.session_state.model._add_intrinsic_anisotropy(
                        C_initial, background
                    )
                    
                    # Apply fluid substitution
                    C_saturated = st.session_state.model._brown_korringa_substitution(
                        C_initial, background, fluid_props
                    )
                    
                    # Calculate properties
                    rho_sat = rho * (1 - porosity) + rho_fluid * porosity
                    
                    Vp_initial = np.sqrt(C_initial[2, 2] / rho)
                    Vs_initial = np.sqrt(C_initial[5, 5] / rho)
                    Vp_sat = np.sqrt(C_saturated[2, 2] / rho_sat)
                    Vs_sat = np.sqrt(C_saturated[5, 5] / rho_sat)
                    
                    # Calculate anisotropy parameters
                    epsilon_initial = (C_initial[0, 0] - C_initial[2, 2]) / (2 * C_initial[2, 2])
                    gamma_initial = (C_initial[5, 5] - C_initial[4, 4]) / (2 * C_initial[4, 4])
                    
                    epsilon_sat = (C_saturated[0, 0] - C_saturated[2, 2]) / (2 * C_saturated[2, 2])
                    gamma_sat = (C_saturated[5, 5] - C_saturated[4, 4]) / (2 * C_saturated[4, 4])
                    
                    # Display results
                    st.subheader("Results")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Initial (Dry)**")
                        st.write(f"Vp: {Vp_initial:.0f} m/s")
                        st.write(f"Vs: {Vs_initial:.0f} m/s")
                        st.write(f"ρ: {rho:.0f} kg/m³")
                        st.write(f"ε: {epsilon_initial:.4f}")
                        st.write(f"γ: {gamma_initial:.4f}")
                    
                    with col_b:
                        st.markdown(f"**Saturated ({fluid_type})**")
                        st.write(f"Vp: {Vp_sat:.0f} m/s")
                        st.write(f"Vs: {Vs_sat:.0f} m/s")
                        st.write(f"ρ: {rho_sat:.0f} kg/m³")
                        st.write(f"ε: {epsilon_sat:.4f}")
                        st.write(f"γ: {gamma_sat:.4f}")
                    
                    # Calculate changes
                    st.subheader("Changes")
                    
                    dVp = ((Vp_sat - Vp_initial) / Vp_initial) * 100
                    dVs = ((Vs_sat - Vs_initial) / Vs_initial) * 100
                    drho = ((rho_sat - rho) / rho) * 100
                    depsilon = ((epsilon_sat - epsilon_initial) / epsilon_initial) * 100
                    dgamma = ((gamma_sat - gamma_initial) / gamma_initial) * 100
                    
                    st.write(f"ΔVp: {dVp:+.1f}%")
                    st.write(f"ΔVs: {dVs:+.1f}%")
                    st.write(f"Δρ: {drho:+.1f}%")
                    st.write(f"Δε: {depsilon:+.1f}%")
                    st.write(f"Δγ: {dgamma:+.1f}%")
                    
                    # Plot comparison
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Velocity comparison
                    ax1 = axes[0]
                    categories = ['Vp', 'Vs']
                    initial_vals = [Vp_initial/1000, Vs_initial/1000]
                    sat_vals = [Vp_sat/1000, Vs_sat/1000]
                    
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    ax1.bar(x - width/2, initial_vals, width, label='Initial', color='blue')
                    ax1.bar(x + width/2, sat_vals, width, label=f'Saturated ({fluid_type})', color='red')
                    
                    ax1.set_xlabel('Property')
                    ax1.set_ylabel('Velocity (km/s)')
                    ax1.set_title('Velocity Comparison')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels(categories)
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Anisotropy comparison
                    ax2 = axes[1]
                    ani_categories = ['ε', 'γ']
                    initial_ani = [epsilon_initial, gamma_initial]
                    sat_ani = [epsilon_sat, gamma_sat]
                    
                    x = np.arange(len(ani_categories))
                    
                    ax2.bar(x - width/2, initial_ani, width, label='Initial', color='blue')
                    ax2.bar(x + width/2, sat_ani, width, label=f'Saturated ({fluid_type})', color='red')
                    
                    ax2.set_xlabel('Parameter')
                    ax2.set_ylabel('Value')
                    ax2.set_title('Anisotropy Comparison')
                    ax2.set_xticks(x)
                    ax2.set_xticklabels(ani_categories)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
    
    elif app_mode == "Inversion":
        st.header("Bayesian Fracture Inversion")
        
        st.warning("Inversion module under development")
        st.info("""
        Planned features:
        1. Markov Chain Monte Carlo (MCMC) sampling
        2. Hamiltonian Monte Carlo for efficiency
        3. Parallel tempering for multi-modal distributions
        4. Hierarchical Bayesian models
        5. Machine learning-assisted proposals
        6. Real-time convergence diagnostics
        """)
        
        # Placeholder for inversion interface
        st.subheader("Inversion Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_chains = st.slider("Number of MCMC chains", 1, 8, 4)
            n_iterations = st.slider("Iterations per chain", 1000, 10000, 5000)
            burn_in = st.slider("Burn-in samples", 100, 2000, 1000)
            
        with col2:
            inversion_method = st.selectbox(
                "Inversion Method",
                ["MCMC", "HMC", "NUTS", "Variational Inference"]
            )
            
            include_uncertainty = st.checkbox("Include parameter uncertainty", True)
            hierarchical = st.checkbox("Hierarchical model", False)
        
        if st.button("Run Inversion (Demo)", type="primary"):
            st.info("Running demo inversion...")
            
            # Generate synthetic posterior for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            # Fracture density posterior
            fracture_density_mean = 0.15
            fracture_density_std = 0.05
            fd_samples = np.random.normal(fracture_density_mean, fracture_density_std, n_samples)
            fd_samples = np.clip(fd_samples, 0.01, 0.3)
            
            # Orientation posterior (von Mises)
            orientation_mean = np.radians(45)
            orientation_kappa = 2.0
            orientation_samples = np.random.vonmises(orientation_mean, orientation_kappa, n_samples)
            
            # Display results
            st.subheader("Posterior Distributions")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Fracture density histogram
            ax1 = axes[0]
            ax1.hist(fd_samples, bins=30, density=True, alpha=0.7, color='steelblue')
            ax1.axvline(fracture_density_mean, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {fracture_density_mean:.3f}')
            ax1.set_xlabel('Fracture Density')
            ax1.set_ylabel('Probability Density')
            ax1.set_title('Fracture Density Posterior')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Orientation histogram
            ax2 = axes[1]
            ax2.hist(np.degrees(orientation_samples), bins=30, density=True, 
                    alpha=0.7, color='green')
            ax2.axvline(np.degrees(orientation_mean), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.degrees(orientation_mean):.1f}°')
            ax2.set_xlabel('Orientation (degrees)')
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Fracture Orientation Posterior')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Summary statistics
            st.subheader("Summary Statistics")
            
            stats_df = pd.DataFrame({
                'Parameter': ['Fracture Density', 'Orientation'],
                'Mean': [np.mean(fd_samples), np.degrees(np.mean(orientation_samples))],
                'Std': [np.std(fd_samples), np.degrees(np.std(orientation_samples))],
                '2.5%': [np.percentile(fd_samples, 2.5), 
                        np.degrees(np.percentile(orientation_samples, 2.5))],
                '97.5%': [np.percentile(fd_samples, 97.5),
                         np.degrees(np.percentile(orientation_samples, 97.5))]
            })
            
            st.dataframe(stats_df.style.format({
                'Mean': '{:.3f}',
                'Std': '{:.3f}',
                '2.5%': '{:.3f}',
                '97.5%': '{:.3f}'
            }))


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run the enhanced AVAz analysis"""
    
    # Create Streamlit app
    create_streamlit_app()


if __name__ == "__main__":
    main()
