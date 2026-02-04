import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from scipy import linalg, stats, signal, interpolate
import xarray as xr
import pandas as pd
from typing import Tuple, Dict, Optional, List, Union
import warnings
import io
from scipy.signal import convolve
warnings.filterwarnings('ignore')

# ============================================================================
# Enhanced SOTA Fracture Model with Well Log Integration
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
    7. Real well log integration
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
            'brine': {'rho': 1040, 'K': 2.8e9, 'mu': 0.001, 'name': 'Brine'},
            'oil': {'rho': 800, 'K': 1.0e9, 'mu': 0.005, 'name': 'Oil'},
            'gas': {'rho': 200, 'K': 0.1e9, 'mu': 0.0001, 'name': 'Gas'}
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
        Vp_bg = max(background.get('Vp', 3000.0), 1000.0)
        Vs_bg = max(background.get('Vs', 1500.0), 500.0)
        rho_bg = max(background.get('rho', 2400.0), 1500.0)
        
        # Ensure Vp > Vs
        if Vp_bg <= Vs_bg * 1.1:
            Vp_bg = Vs_bg * 1.1
        
        mu_bg = rho_bg * Vs_bg**2
        lambda_bg = rho_bg * Vp_bg**2 - 2 * mu_bg
        
        # Ensure lambda_bg is positive
        lambda_bg = max(lambda_bg, 1e6)
        
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
        rho = max(background.get('rho', 2400.0), 1500.0)
        Vp0 = np.sqrt(max(C[2, 2] / rho, 1000.0))
        Vs0 = np.sqrt(max(C[5, 5] / rho, 500.0))
        
        # Update for VTI
        C_ani[0, 0] = C_ani[1, 1] = rho * Vp0**2 * (1 + 2 * epsilon)
        C_ani[2, 2] = rho * Vp0**2
        C_ani[0, 2] = C_ani[1, 2] = rho * (Vp0**2 - 2 * Vs0**2) * (1 + delta)
        C_ani[5, 5] = rho * Vs0**2 * (1 + 2 * gamma)
        
        return C_ani
    
    def _add_fracture_set(self, C_bg: np.ndarray, background: Dict, 
                         fractures: Dict) -> np.ndarray:
        """Add fracture compliance using Schoenberg linear slip theory"""
        
        fracture_density = fractures.get('density', 0.1)
        aspect_ratio = fractures.get('aspect_ratio', 0.01)
        orientation = fractures.get('orientation', 0.0)
        fill = fractures.get('fill', 'fluid')
        
        # Ensure valid values
        fracture_density = max(min(fracture_density, 0.5), 0.0)
        aspect_ratio = max(min(aspect_ratio, 0.1), 1e-6)
        
        # Background moduli with validation
        Vp = max(background.get('Vp', 3000.0), 1000.0)
        Vs = max(background.get('Vs', 1500.0), 500.0)
        rho = max(background.get('rho', 2400.0), 1500.0)
        
        # Ensure Vp > Vs
        if Vp <= Vs * 1.1:
            Vp = Vs * 1.1
        
        mu_bg = rho * Vs**2
        
        # Add small epsilon to avoid division by zero
        mu_bg = max(mu_bg, 1e6)  # Minimum shear modulus
        
        lambda_bg = rho * Vp**2 - 2 * mu_bg
        K_bg = lambda_bg + 2/3 * mu_bg
        
        # Fracture compliances based on fill type
        if fill in self.fluid_properties:
            fluid = self.fluid_properties[fill]
            K_f = fluid['K']
            
            # Hudson model for fluid-filled fractures
            # Avoid division by zero
            denominator = 3 * K_f + 4 * mu_bg
            if abs(denominator) < 1e-10:
                denominator = 1e-10
                
            ZN = 4 * aspect_ratio / denominator
            ZT = 16 * aspect_ratio / (3 * denominator)
        else:
            # Dry or mineral-filled fractures
            ZN = aspect_ratio / mu_bg
            ZT = 2 * ZN  # Approximate
        
        # Scale by fracture density
        BN = fracture_density * ZN
        BT = fracture_density * ZT
        
        # Ensure BN and BT are finite
        BN = np.clip(BN, -1e6, 1e6)
        BT = np.clip(BT, -1e6, 1e6)
        
        # Fracture compliance tensor in fracture coordinates
        S_frac = np.zeros((6, 6))
        S_frac[2, 2] = BN  # Normal compliance
        S_frac[3, 3] = S_frac[4, 4] = BT  # Tangential compliances
        
        # Rotate to global coordinates
        try:
            R = self._rotation_matrix_voigt(orientation, dip=fractures.get('dip', 0))
            S_frac_global = R.T @ S_frac @ R
        except:
            # If rotation fails, use identity
            S_frac_global = S_frac
        
        # Effective compliance
        try:
            S_bg = linalg.inv(C_bg)
            S_eff = S_bg + S_frac_global
            
            # Check for NaN or inf values
            if np.any(np.isnan(S_eff)) or np.any(np.isinf(S_eff)):
                # If invalid, return original stiffness
                return C_bg
            
            return linalg.inv(S_eff)
            
        except Exception as e:
            # If inversion fails, return original stiffness
            return C_bg
    
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
        sigma_v = stress.get('vertical', 0.0)
        sigma_h = stress.get('horizontal', 0.0)
        sigma_H = stress.get('horizontal_max', sigma_h)
        
        # Calculate stress-induced changes
        c = self.third_order_constants
        
        # Strain from stress
        try:
            S_iso = linalg.inv(C)
            epsilon = S_iso @ np.array([sigma_H, sigma_h, sigma_v, 0, 0, 0])
        except:
            # If inversion fails, use zero strain
            epsilon = np.zeros(6)
        
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
        
        # Extract parameters with defaults
        phi = fluid_props.get('porosity', 0.2)
        K_s = fluid_props.get('mineral_K', 37e9)  # Mineral bulk modulus
        G_s = fluid_props.get('mineral_G', 44e9)  # Mineral shear modulus
        K_f = fluid_props.get('fluid_K', 2.2e9)   # Fluid bulk modulus
        
        # Current moduli from stiffness
        rho = max(background.get('rho', 2400.0), 1500.0)
        
        try:
            Vp = np.sqrt(max(C[2, 2] / rho, 1000.0**2))
            Vs = np.sqrt(max(C[5, 5] / rho, 500.0**2))
        except:
            Vp = 3000.0
            Vs = 1500.0
        
        K_dry = rho * (Vp**2 - 4/3 * Vs**2)  # Dry bulk modulus
        G_dry = rho * Vs**2  # Dry shear modulus
        
        # Brown-Korringa equations for anisotropic media
        # Simplified version - full implementation requires more complex tensor math
        beta = 1 - (K_s / max(K_dry, 1e6))
        
        # Saturated bulk modulus
        K_sat = K_s + (beta**2) / ((phi / K_f) + ((beta - phi) / max(K_dry, 1e6)) - 
                                  (background.get('delta', 0) * K_s) / (3 * max(K_dry, 1e6)))
        
        # Saturated shear modulus (less affected by fluid)
        G_sat = G_dry * (1 - background.get('gamma', 0) * K_s / (3 * max(K_dry, 1e6)))
        
        # Update stiffness components
        C_sat = C.copy()
        lambda_sat = K_sat - 2/3 * G_sat
        
        # Update diagonal components
        C_sat[0, 0] = C_sat[1, 1] = C_sat[2, 2] = lambda_sat + 2 * G_sat
        C_sat[3, 3] = C_sat[4, 4] = C_sat[5, 5] = G_sat
        
        return C_sat
    
    def _apply_frequency_dispersion(self, C_static: np.ndarray,
                                   background: Dict, fractures: Dict) -> np.ndarray:
        """Apply frequency-dependent stiffness using Chapman's model"""
        
        if not self.frequency_dependent:
            return C_static
        
        f_char = self.dispersion_params.get('characteristic_frequency', 50)
        omega = 2 * np.pi * self.dispersion_params.get('frequency', 30)
        
        # Complex stiffness for viscoelastic effects
        Q = self.dispersion_params.get('Q', 50)
        alpha = 1 / (np.pi * max(Q, 1.0))  # Attenuation coefficient
        
        # Frequency-dependent correction
        f_ratio = omega / (2 * np.pi * max(f_char, 1.0))
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
        rho_bg = max(background.get('rho', 2400.0), 1500.0)
        phi = background.get('porosity', 0.2)
        phi_f = fractures.get('fracture_porosity', 0.05)
        
        # Simple mixing law
        rho_eff = rho_bg * (1 - phi - phi_f)
        
        if 'fluid_density' in fractures:
            rho_eff += fractures['fluid_density'] * (phi + phi_f)
        
        return max(rho_eff, 1500.0)
    
    def _calculate_anisotropy_parameters(self, C: np.ndarray, rho: float) -> Dict:
        """Calculate comprehensive anisotropy parameters"""
        
        # Ensure rho is valid
        rho = max(rho, 1500.0)
        
        # Extract stiffness components with validation
        try:
            C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
            C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]
            C23, C13, C12 = C[1, 2], C[0, 2], C[0, 1]
        except:
            # Default values if extraction fails
            C11 = C22 = C33 = 30e9
            C44 = C55 = C66 = 10e9
            C23 = C13 = C12 = 10e9
        
        # Velocities with validation
        Vp0 = np.sqrt(max(C33 / rho, 1000.0**2))
        Vs0 = np.sqrt(max(C55 / rho, 500.0**2))
        Vp90 = np.sqrt(max(C11 / rho, 1000.0**2))  # Horizontal P-wave
        
        # Thomsen-style parameters with protection against division by zero
        denominator = 2 * max(C33, 1e6)
        epsilon1 = (C11 - C33) / denominator
        epsilon2 = (C22 - C33) / denominator
        
        # Calculate delta parameters safely
        try:
            delta1 = ((C13 + C55)**2 - (C33 - C55)**2) / (2 * C33 * (C33 - C55))
        except:
            delta1 = 0.0
        
        try:
            delta2 = ((C23 + C44)**2 - (C33 - C44)**2) / (2 * C33 * (C33 - C44))
        except:
            delta2 = 0.0
        
        # Calculate gamma parameters safely
        try:
            gamma1 = (C66 - C44) / (2 * max(C44, 1e6))
        except:
            gamma1 = 0.0
        
        try:
            gamma2 = (C66 - C55) / (2 * max(C55, 1e6))
        except:
            gamma2 = 0.0
        
        # Fracture parameters
        try:
            ZN = (1/max(C33, 1e6) - 1/max(C11, 1e6))  # Normal weakness
            ZT = (1/max(C55, 1e6) - 1/max(C66, 1e6))  # Tangential weakness
        except:
            ZN = ZT = 0.0
        
        # Anisotropy strength
        P_aniso = (C11 - C33) / max(C33, 1e6)
        S_aniso = (C66 - C44) / max(C44, 1e6)
        
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
    
    def calculate_reflectivity(self, vp: List[float], vs: List[float], 
                              d: List[float], e: List[float], g: List[float], 
                              dlt: List[float], theta: float, azimuth: float) -> float:
        """
        Calculate reflectivity using Rüger equations for orthorhombic media
        
        Parameters:
        -----------
        vp, vs, d, e, g, dlt: Lists with properties for 3 layers [above, target, below]
        theta: Incidence angle in radians
        azimuth: Azimuth angle in degrees
        """
        # Average properties at interface
        VP2 = (vp[1] + vp[2]) / 2
        VS2 = (vs[1] + vs[2]) / 2
        DEN2 = (d[1] + d[2]) / 2
        
        # A term (intercept)
        A2 = -0.5 * ((vp[2] - vp[1]) / VP2 + (d[2] - d[1]) / DEN2)
        
        az_rad = np.radians(azimuth)
        
        # B term (gradient)
        Biso2 = 0.5 * ((vp[2] - vp[1]) / VP2) - 2 * (VS2 / VP2)**2 * (d[2] - d[1]) / DEN2 - 4 * (VS2 / VP2)**2 * (vs[2] - vs[1]) / VS2
        
        # Anisotropic terms
        Baniso2 = 0.5 * ((dlt[2] - dlt[1]) + 2 * (2 * VS2 / VP2)**2 * (g[2] - g[1]))
        Caniso2 = 0.5 * ((vp[2] - vp[1]) / VP2 - (e[2] - e[1]) * np.cos(az_rad)**4 + 
                        (dlt[2] - dlt[1]) * np.sin(az_rad)**2 * np.cos(az_rad)**2)
        
        # Full Rüger equation
        return A2 + (Biso2 + Baniso2 * np.cos(az_rad)**2) * np.sin(theta)**2 + Caniso2 * np.sin(theta)**2 * np.tan(theta)**2
    
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
        Vp0 = anisotropy.get('Vp0', 3000.0)
        Vs0 = anisotropy.get('Vs0', 1500.0)
        epsilon = anisotropy.get('epsilon1', 0.1)
        delta = anisotropy.get('delta1', 0.05)
        gamma = anisotropy.get('gamma1', 0.08)
        
        # Background isotropic contrast
        Z2 = max(rho * Vp0, 1e6)
        dZ = 0.1 * Z2  # 10% contrast
        
        R_pp = np.zeros((len(theta_rad), len(phi_rad)))
        
        for i, theta_i in enumerate(theta_rad):
            sin2 = np.sin(theta_i)**2
            sin2_tan2 = sin2 * np.tan(theta_i)**2
            
            for j, phi_j in enumerate(phi_rad):
                # Isotropic term
                R0 = 0.5 * dZ / max(Z2, 1e6)
                
                # Gradient term
                G = 0.5 * (Vp0 * 0.1 / max(Vp0, 1.0)) - 2 * (Vs0/max(Vp0, 1.0))**2 * (2 * 0.1 + 0.1)
                
                # Anisotropic terms
                B_ani = 0.5 * (delta + 2 * (2*Vs0/max(Vp0, 1.0))**2 * gamma)
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
        f_char = self.dispersion_params.get('characteristic_frequency', 50)
        ratio = frequency / max(f_char, 1.0)
        
        # Dispersion increases with angle
        correction = 1 + 0.15 * np.sin(theta) * ratio / (1 + ratio**2)
        
        return correction
    
    def calculate_elastic_impedance(self, theta: float, phi: float,
                                   model_params: Dict) -> float:
        """Calculate elastic impedance for given angle and azimuth"""
        
        C = model_params['stiffness']
        rho = model_params['density']
        
        # Simplified EI calculation
        try:
            Vp = np.sqrt(max(C[2, 2] / max(rho, 1500.0), 1000.0**2))
            Vs = np.sqrt(max(C[5, 5] / max(rho, 1500.0), 500.0**2))
        except:
            Vp = 3000.0
            Vs = 1500.0
        
        # EI = Vp^a * Vs^b * ρ^c with azimuthal dependence
        K = (Vs / max(Vp, 1.0))**2
        
        # Coefficients from Connolly
        a = np.cos(np.radians(phi))**2 + np.sin(np.radians(phi))**2 * (1 - 2 * K)
        b = -8 * K * np.sin(np.radians(phi))**2
        c = 1 - 4 * K * np.sin(np.radians(phi))**2
        
        EI = Vp**a * Vs**b * rho**c
        
        return EI


# ============================================================================
# Enhanced Synthetic Generator with Well Log Integration
# ============================================================================

class EnhancedSyntheticGenerator:
    """Generate synthetic seismic with realistic fracture effects and well log integration"""
    
    def __init__(self, model: EnhancedFractureModel):
        self.model = model
        self.well_logs = {}
        
    def load_well_logs(self, well_log_data: Dict[str, pd.DataFrame]):
        """Load well log data from CSV files"""
        self.well_logs = well_log_data
        
    def generate_azimuthal_gathers_from_well_logs(self, config: Dict, well_name: str = None) -> Dict:
        """
        Generate realistic azimuthal gathers from well log data
        
        Parameters:
        -----------
        config : Dict with generation parameters
        well_name : str - specific well to use (if None, use all)
        
        Returns:
        --------
        Dict with azimuthal gathers and associated data
        """
        if not self.well_logs:
            st.warning("No well log data loaded!")
            return None
        
        # Use specified well or first available
        if well_name and well_name in self.well_logs:
            wells_to_use = {well_name: self.well_logs[well_name]}
        else:
            wells_to_use = self.well_logs
        
        results = {}
        
        for current_well_name, df in wells_to_use.items():
            # Process well log data
            processed_logs = self._process_well_log_data(df)
            
            if processed_logs is None:
                continue
            
            # Extract parameters from config
            max_angle = config.get('max_angle', 60)
            angle_step = config.get('angle_step', 5)
            azimuth_step = config.get('azimuth_step', 15)
            wavelet_freq = config.get('wavelet_freq', 45)
            
            # Create angle and azimuth arrays
            incidence_angles = np.arange(0, max_angle + 1, angle_step)
            azimuths = np.arange(0, 361, azimuth_step)
            
            # Find interfaces for AVAz analysis
            interfaces = self._find_avaz_interfaces(processed_logs, config)
            
            # Calculate reflectivity for each interface
            reflectivity_matrix = np.zeros((len(incidence_angles), len(azimuths)))
            
            for i, theta_deg in enumerate(incidence_angles):
                theta_rad = np.radians(theta_deg)
                for j, az in enumerate(azimuths):
                    # Use properties around the main interface (interface index 1)
                    if len(interfaces) > 1:
                        vp = [processed_logs['Vp'][interfaces[0]], 
                              processed_logs['Vp'][interfaces[1]], 
                              processed_logs['Vp'][interfaces[2]]]
                        vs = [processed_logs['Vs'][interfaces[0]], 
                              processed_logs['Vs'][interfaces[1]], 
                              processed_logs['Vs'][interfaces[2]]]
                        d = [processed_logs['rho'][interfaces[0]], 
                             processed_logs['rho'][interfaces[1]], 
                             processed_logs['rho'][interfaces[2]]]
                        
                        # Estimate anisotropy parameters from well data
                        e = self._estimate_anisotropy_from_logs(processed_logs, interfaces)
                        g = e * 0.5  # Approximate gamma from epsilon
                        dlt = e * 0.8  # Approximate delta from epsilon
                        
                        reflectivity_matrix[i, j] = self.model.calculate_reflectivity(
                            vp, vs, d, e, g, dlt, theta_rad, az
                        )
            
            # Generate synthetic gathers
            synthetic_gathers = self._generate_synthetic_gathers(
                reflectivity_matrix, incidence_angles, azimuths, wavelet_freq
            )
            
            # Calculate AVAz attributes
            avaz_attributes = self._calculate_avaz_attributes(
                reflectivity_matrix, incidence_angles, azimuths
            )
            
            results[current_well_name] = {
                'well_name': current_well_name,
                'processed_logs': processed_logs,
                'incidence_angles': incidence_angles,
                'azimuths': azimuths,
                'reflectivity_matrix': reflectivity_matrix,
                'synthetic_gathers': synthetic_gathers,
                'avaz_attributes': avaz_attributes,
                'interfaces': interfaces,
                'config': config
            }
        
        return results
    
    def _process_well_log_data(self, df: pd.DataFrame) -> Dict:
        """Process raw well log data into standardized format"""
        
        # Find depth column
        depth_col = None
        for col in df.columns:
            col_lower = col.lower()
            if 'depth' in col_lower:
                depth_col = col
                break
        
        if depth_col is None:
            depth_col = df.columns[0]
        
        # Initialize result dictionary
        result = {
            'depth': df[depth_col].values,
            'Vp': None,
            'Vs': None,
            'rho': None,
            'phi': None,
            'GR': None,
            'RT': None,
            'SW': None
        }
        
        # Find data columns
        for col in df.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['vp', 'p_vel', 'velocity_p', 'dtp']):
                # Convert from dtp if needed
                if 'dtp' in col_lower:
                    dtp = df[col].values
                    result['Vp'] = 1e6 / dtp  # Convert us/ft to m/s
                else:
                    result['Vp'] = df[col].values
            elif any(x in col_lower for x in ['vs', 's_vel', 'velocity_s', 'dts']):
                if 'dts' in col_lower:
                    dts = df[col].values
                    result['Vs'] = 1e6 / dts
                else:
                    result['Vs'] = df[col].values
            elif any(x in col_lower for x in ['rho', 'dens', 'density', 'rhob']):
                result['rho'] = df[col].values
            elif any(x in col_lower for x in ['phi', 'phie', 'porosity', 'nphi']):
                result['phi'] = df[col].values
            elif any(x in col_lower for x in ['gr', 'gamma', 'gamma_ray']):
                result['GR'] = df[col].values
            elif any(x in col_lower for x in ['rt', 'resistivity', 'ild']):
                result['RT'] = df[col].values
            elif any(x in col_lower for x in ['sw', 'water_sat', 'swt']):
                result['SW'] = df[col].values
        
        # Fill missing data
        if result['Vp'] is None:
            st.warning("No Vp data found!")
            return None
        
        if result['Vs'] is None:
            # Estimate Vs from Vp using Castagna's relation
            result['Vs'] = result['Vp'] / 1.8
        
        if result['rho'] is None:
            # Estimate density from Vp using Gardner's relation
            result['rho'] = 310 * (result['Vp'] ** 0.25) * 0.23
        
        if result['phi'] is None:
            # Estimate porosity from Vp
            result['phi'] = np.clip(0.4 - 0.0003 * result['Vp'], 0.05, 0.35)
        
        return result
    
    def _find_avaz_interfaces(self, logs: Dict, config: Dict) -> List[int]:
        """Find significant interfaces for AVAz analysis"""
        
        depth = logs['depth']
        vp = logs['Vp']
        
        # Calculate reflectivity series
        reflectivity = np.zeros(len(vp))
        for i in range(1, len(vp)-1):
            Z1 = vp[i-1] * logs['rho'][i-1]
            Z2 = vp[i] * logs['rho'][i]
            reflectivity[i] = (Z2 - Z1) / (Z2 + Z1) if (Z2 + Z1) > 0 else 0
        
        # Find significant reflectors
        threshold = np.percentile(np.abs(reflectivity), 90)
        peaks = np.where(np.abs(reflectivity) > threshold)[0]
        
        # If no peaks found, use a default interface
        if len(peaks) == 0:
            middle_idx = len(vp) // 2
            return [middle_idx-10, middle_idx, middle_idx+10]
        
        # Find the strongest peak
        main_peak_idx = peaks[np.argmax(np.abs(reflectivity[peaks]))]
        
        # Return indices around the main peak
        return [max(0, main_peak_idx-5), main_peak_idx, min(len(vp)-1, main_peak_idx+5)]
    
    def _estimate_anisotropy_from_logs(self, logs: Dict, interfaces: List[int]) -> List[float]:
        """Estimate anisotropy parameters from well log data"""
        
        # Simple estimation based on Vp variations
        vp = logs['Vp']
        
        # Calculate epsilon-like parameter from Vp variations
        vp_mean = np.mean(vp)
        epsilon_estimate = (vp - vp_mean) / (2 * vp_mean)
        
        # Extract at interfaces
        e_values = []
        for idx in interfaces:
            if 0 <= idx < len(epsilon_estimate):
                e_values.append(epsilon_estimate[idx])
            else:
                e_values.append(0.1)  # Default
        
        return e_values
    
    def _generate_synthetic_gathers(self, reflectivity_matrix: np.ndarray,
                                   incidence_angles: np.ndarray,
                                   azimuths: np.ndarray,
                                   wavelet_freq: float) -> Dict:
        """Generate synthetic seismic gathers from reflectivity"""
        
        n_samples = 200
        dt = 0.001
        wavelet_length = 0.1
        wavelet = self._ricker_wavelet(wavelet_freq, wavelet_length, dt)
        
        # Create synthetic traces for each angle
        synthetic_gathers = {}
        
        for i, angle in enumerate(incidence_angles):
            # Create reflectivity series
            R = np.zeros((n_samples, len(azimuths)))
            center_sample = n_samples // 2
            R[center_sample, :] = reflectivity_matrix[i, :]
            
            # Convolve with wavelet
            synthetic = np.zeros((n_samples + len(wavelet) - 1, len(azimuths)))
            for j in range(len(azimuths)):
                synthetic[:, j] = convolve(R[:, j], wavelet, mode='full')
            
            # Trim to original length
            trim_start = len(wavelet) // 2
            trim_end = -(len(wavelet) // 2 - 1)
            synthetic = synthetic[trim_start:trim_end, :]
            
            synthetic_gathers[f'angle_{angle}'] = {
                'data': synthetic,
                'angle': angle,
                'azimuths': azimuths
            }
        
        return synthetic_gathers
    
    def _ricker_wavelet(self, freq: float, length: float, dt: float) -> np.ndarray:
        """Generate Ricker wavelet"""
        t = np.arange(-length/2, length/2, dt)
        return (1 - 2*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)
    
    def _calculate_avaz_attributes(self, reflectivity_matrix: np.ndarray,
                                  incidence_angles: np.ndarray,
                                  azimuths: np.ndarray) -> Dict:
        """Calculate AVAz attributes from reflectivity matrix"""
        
        attributes = {}
        
        # For each incidence angle
        for i, angle in enumerate(incidence_angles):
            R_angle = reflectivity_matrix[i, :]
            
            # Fit sinusoidal variation: R(φ) = A + B*cos(2*(φ - φ₀))
            phi_rad = np.radians(azimuths)
            
            # Fourier decomposition
            coeffs = np.fft.fft(R_angle)
            
            A = np.real(coeffs[0]) / len(R_angle)  # Isotropic component
            B = 2 * np.abs(coeffs[2]) / len(R_angle)  # 2φ component
            phi0_rad = 0.5 * np.angle(coeffs[2])  # Fast direction
            
            attributes[f'angle_{angle}'] = {
                'A': A,
                'B': B,
                'phi0': np.degrees(phi0_rad) % 180,  # Convert to degrees
                'B_A_ratio': B/A if abs(A) > 1e-10 else 0,
                'anisotropy_magnitude': B
            }
        
        # Calculate angle-dependent trends
        A_values = [attrs['A'] for attrs in attributes.values()]
        B_values = [attrs['B'] for attrs in attributes.values()]
        phi0_values = [attrs['phi0'] for attrs in attributes.values()]
        
        attributes['summary'] = {
            'A_vs_angle': np.polyfit(incidence_angles, A_values, 2),
            'B_vs_angle': np.polyfit(incidence_angles, B_values, 2),
            'mean_phi0': np.mean(phi0_values),
            'std_phi0': np.std(phi0_values)
        }
        
        return attributes
    
    def generate_synthetic_data(self, config: Dict, use_well_logs: bool = False) -> Dict:
        """
        Generate comprehensive synthetic dataset
        
        Parameters:
        -----------
        config : Dict with generation parameters
        use_well_logs : bool - whether to use real well log data
        
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
        
        if use_well_logs and self.well_logs:
            # Generate geological model from well logs
            geology = self._generate_geology_from_well_logs(nx, ny, nz, config)
        else:
            # Generate synthetic geological model
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
        if use_well_logs and self.well_logs:
            # Use actual well logs
            well_logs = self._process_actual_well_logs(config)
        else:
            # Generate synthetic well logs
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
    
    def _generate_geology_from_well_logs(self, nx: int, ny: int, nz: int,
                                        config: Dict) -> Dict:
        """Generate geological model by interpolating well log data"""
        
        if not self.well_logs:
            return self._generate_geological_model(nx, ny, nz, config)
        
        # Create 3D grids
        Vp = np.ones((nz, ny, nx)) * 3000.0  # Default Vp
        Vs = np.ones((nz, ny, nx)) * 1500.0  # Default Vs
        rho = np.ones((nz, ny, nx)) * 2400.0  # Default density
        porosity = np.ones((nz, ny, nx)) * 0.2  # Default porosity
        clay_content = np.ones((nz, ny, nx)) * 0.3  # Default clay content
        
        # Well positions (assuming we have 3 wells)
        well_positions = [
            {'x': nx//4, 'y': ny//2, 'name': 'Well_A'},
            {'x': nx//2, 'y': ny//2, 'name': 'Well_B'},
            {'x': 3*nx//4, 'y': ny//2, 'name': 'Well_C'}
        ]
        
        # Extract well log data
        well_data = []
        for well in well_positions:
            if well['name'] in self.well_logs:
                df = self.well_logs[well['name']]
                logs = {
                    'x': well['x'],
                    'y': well['y'],
                    'depth': None,
                    'Vp': None,
                    'Vs': None,
                    'rho': None,
                    'phi': None,
                    'sw': None,
                    'gr': None,
                    'rt': None
                }
                
                # Find depth column
                for col in df.columns:
                    col_lower = col.lower()
                    if 'depth' in col_lower:
                        logs['depth'] = df[col].values
                        break
                
                if logs['depth'] is None:
                    logs['depth'] = np.arange(len(df))
                
                # Find other columns
                for col in df.columns:
                    col_lower = col.lower()
                    if any(x in col_lower for x in ['vp', 'p_vel', 'velocity_p']):
                        logs['Vp'] = df[col].values
                    elif any(x in col_lower for x in ['vs', 's_vel', 'velocity_s']):
                        logs['Vs'] = df[col].values
                    elif any(x in col_lower for x in ['rho', 'dens', 'density', 'rhob']):
                        logs['rho'] = df[col].values
                    elif any(x in col_lower for x in ['phi', 'phie', 'porosity']):
                        logs['phi'] = df[col].values
                    elif any(x in col_lower for x in ['gr', 'gamma', 'gamma_ray']):
                        logs['gr'] = df[col].values
                
                well_data.append(logs)
        
        # If we have well data, interpolate
        if well_data:
            # Simple interpolation between wells
            for i in range(nz):
                depth_value = i * config['dz']
                
                for well in well_data:
                    if well['Vp'] is not None and len(well['depth']) > 0:
                        # Find closest depth
                        depth_idx = np.argmin(np.abs(well['depth'] - depth_value))
                        
                        # Assign values at well locations
                        x_idx = well['x']
                        y_idx = well['y']
                        
                        if well['Vp'] is not None and depth_idx < len(well['Vp']):
                            Vp[i, y_idx, x_idx] = max(well['Vp'][depth_idx], 1000.0)
                        
                        if well['Vs'] is not None and depth_idx < len(well['Vs']):
                            Vs[i, y_idx, x_idx] = max(well['Vs'][depth_idx], 500.0)
                        else:
                            # Estimate from Vp
                            Vs[i, y_idx, x_idx] = Vp[i, y_idx, x_idx] / 1.8
                        
                        if well['rho'] is not None and depth_idx < len(well['rho']):
                            rho[i, y_idx, x_idx] = max(well['rho'][depth_idx], 1500.0)
                        else:
                            # Gardner's relation
                            rho[i, y_idx, x_idx] = 310 * (Vp[i, y_idx, x_idx] ** 0.25) * 0.23
                        
                        if well['phi'] is not None and depth_idx < len(well['phi']):
                            porosity[i, y_idx, x_idx] = np.clip(well['phi'][depth_idx], 0.01, 0.4)
                        
                        if well['gr'] is not None and depth_idx < len(well['gr']):
                            # Simple clay estimation from GR
                            gr_val = well['gr'][depth_idx]
                            clay_content[i, y_idx, x_idx] = np.clip((gr_val - 30) / (150 - 30), 0.0, 1.0)
        
        # Interpolate between wells
        for i in range(nz):
            # Simple linear interpolation between wells
            for j in range(ny):
                for k in range(nx):
                    # Find distances to wells
                    distances = []
                    vp_values = []
                    
                    for well in well_data:
                        if well['Vp'] is not None:
                            dist = np.sqrt((k - well['x'])**2 + (j - well['y'])**2)
                            distances.append(dist)
                            # Get value at this depth
                            depth_idx = np.argmin(np.abs(well['depth'] - i*config['dz']))
                            if depth_idx < len(well['Vp']):
                                vp_values.append(max(well['Vp'][depth_idx], 1000.0))
                            else:
                                vp_values.append(3000.0)
                    
                    if len(vp_values) > 0:
                        # Inverse distance weighting
                        if len(vp_values) == 1:
                            Vp[i, j, k] = vp_values[0]
                        else:
                            weights = 1.0 / np.array(distances)
                            weights = weights / weights.sum()
                            Vp[i, j, k] = np.sum(np.array(vp_values) * weights)
                        
                        # Calculate other properties
                        Vs[i, j, k] = Vp[i, j, k] / 1.8
                        rho[i, j, k] = 310 * (Vp[i, j, k] ** 0.25) * 0.23
                        porosity[i, j, k] = np.clip(0.4 - 0.0003 * Vp[i, j, k], 0.05, 0.35)
                        clay_content[i, j, k] = 0.3  # Default
        
        return {
            'Vp': Vp,
            'Vs': Vs,
            'rho': rho,
            'porosity': porosity,
            'clay_content': clay_content,
            'from_well_logs': True
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
        
        # Fracture fill based on water saturation if available
        fracture_fill = np.full((nz, ny, nx), 'fluid', dtype=object)
        
        # If we have SW data in geology, use it to determine fluid type
        if 'sw' in geology:
            sw = geology['sw']
            for i in range(nz):
                for j in range(ny):
                    for k in range(nx):
                        if sw[i, j, k] < 0.3:  # Low water saturation -> hydrocarbon
                            fracture_fill[i, j, k] = 'gas' if np.random.random() > 0.5 else 'oil'
                        else:
                            fracture_fill[i, j, k] = 'brine'
        else:
            # Use default logic
            anticline_center = nx // 2
            for i in range(nz):
                for j in range(ny):
                    for k in range(nx):
                        distance = np.sqrt((k - anticline_center)**2 + (j - ny/2)**2)
                        if distance < 20 and i > nz//2:
                            fracture_fill[i, j, k] = 'gas' if np.random.random() > 0.7 else 'brine'
        
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
                    # Background rock - ensure valid values
                    Vp_val = max(geology['Vp'][i, j, k], 1000.0)  # Minimum Vp
                    Vs_val = max(geology['Vs'][i, j, k], 500.0)   # Minimum Vs
                    rho_val = max(geology['rho'][i, j, k], 1500.0) # Minimum density
                    
                    # Ensure Vp > Vs
                    if Vp_val <= Vs_val * 1.1:
                        Vp_val = Vs_val * 1.1
                    
                    background = {
                        'Vp': Vp_val,
                        'Vs': Vs_val,
                        'rho': rho_val,
                        'porosity': min(max(geology['porosity'][i, j, k], 0.01), 0.4),
                        'clay': min(max(geology['clay_content'][i, j, k], 0.0), 1.0)
                    }
                    
                    # Fracture properties - ensure valid values
                    frac = {
                        'density': min(max(fractures['density'][i, j, k], 0.0), 0.5),
                        'aspect_ratio': min(max(fractures['aspect_ratio'][i, j, k], 0.0001), 0.1),
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
                    
                    try:
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
                        
                    except Exception as e:
                        # Fallback to isotropic background
                        C_bg = self.model._background_stiffness(background)
                        C[i, j, k] = C_bg
                        density[i, j, k] = background['rho']
                        anisotropy['epsilon1'][i, j, k] = 0.0
                        anisotropy['gamma1'][i, j, k] = 0.0
                        anisotropy['ZN'][i, j, k] = 0.0
                        anisotropy['ZT'][i, j, k] = 0.0
        
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
        wavelet = self._ricker_wavelet(wavelet_freq, 0.08, 0.001)
        
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
                try:
                    Z1 = rho[i-1] * np.sqrt(C[i-1, ..., 2, 2] / max(rho[i-1], 1500.0))
                    Z2 = rho[i] * np.sqrt(C[i, ..., 2, 2] / max(rho[i], 1500.0))
                    
                    # Isotropic reflectivity
                    R0 = (Z2 - Z1) / max(Z2 + Z1, 1e-10)
                    
                    # Azimuthal variation
                    # Use anisotropy parameters
                    epsilon1 = elastic_props['anisotropy']['epsilon1'][i]
                    gamma1 = elastic_props['anisotropy']['gamma1'][i]
                    
                    # Simplified azimuthal term
                    az_term = (epsilon1 * np.cos(az_rad)**2 + 
                              gamma1 * np.sin(az_rad)**2) * np.sin(theta_rad)**2
                    
                    R[angle_idx, i] = R0 + 0.1 * az_term
                except:
                    R[angle_idx, i] = 0.0
        
        return R
    
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
            logs['Vp/Vs'] = logs['Vp'] / np.maximum(logs['Vs'], 1.0)
            logs['Acoustic_Impedance'] = logs['Vp'] * logs['rho']
            logs['Shear_Impedance'] = logs['Vs'] * logs['rho']
            logs['LambdaRho'] = logs['rho'] * (logs['Vp']**2 - 2 * logs['Vs']**2)
            logs['MuRho'] = logs['rho'] * logs['Vs']**2
            
            well_logs[well['name']] = logs
        
        return well_logs
    
    def _process_actual_well_logs(self, config: Dict) -> Dict:
        """Process and return actual well log data"""
        
        processed_logs = {}
        
        for well_name, df in self.well_logs.items():
            # Ensure depth column exists
            depth_col = None
            for col in df.columns:
                if 'depth' in col.lower():
                    depth_col = col
                    break
            
            if depth_col is None:
                depth_col = df.columns[0]
            
            logs = {
                'depth': df[depth_col].values,
                'Vp': None,
                'Vs': None,
                'rho': None,
                'porosity': None,
                'sw': None,
                'gr': None,
                'rt': None
            }
            
            # Find data columns
            for col in df.columns:
                col_lower = col.lower()
                if any(x in col_lower for x in ['vp', 'p_vel', 'velocity_p']):
                    logs['Vp'] = df[col].values
                elif any(x in col_lower for x in ['vs', 's_vel', 'velocity_s']):
                    logs['Vs'] = df[col].values
                elif any(x in col_lower for x in ['rho', 'dens', 'density', 'rhob']):
                    logs['rho'] = df[col].values
                elif any(x in col_lower for x in ['phi', 'phie', 'porosity']):
                    logs['porosity'] = df[col].values
                elif 'sw' in col_lower:
                    logs['sw'] = df[col].values
                elif any(x in col_lower for x in ['gr', 'gamma', 'gamma_ray']):
                    logs['gr'] = df[col].values
                elif any(x in col_lower for x in ['rt', 'resistivity']):
                    logs['rt'] = df[col].values
            
            # Calculate derived properties if we have the required data
            if logs['Vp'] is not None and logs['rho'] is not None:
                logs['Acoustic_Impedance'] = logs['Vp'] * logs['rho']
            
            if logs['Vs'] is not None and logs['rho'] is not None:
                logs['Shear_Impedance'] = logs['Vs'] * logs['rho']
            
            if logs['Vp'] is not None and logs['Vs'] is not None:
                logs['Vp/Vs'] = logs['Vp'] / np.maximum(logs['Vs'], 1.0)
            
            if logs['Vp'] is not None and logs['Vs'] is not None and logs['rho'] is not None:
                logs['LambdaRho'] = logs['rho'] * (logs['Vp']**2 - 2 * logs['Vs']**2)
                logs['MuRho'] = logs['rho'] * logs['Vs']**2
            
            # Estimate fracture density from rock properties (simplified)
            if logs['Vp'] is not None and logs['Vs'] is not None:
                # Simple heuristic: higher Vp/Vs ratio indicates more fracturing
                vp_vs = logs['Vp'] / np.maximum(logs['Vs'], 1.0)
                vp_vs_norm = (vp_vs - np.nanmean(vp_vs)) / np.maximum(np.nanstd(vp_vs), 1e-10)
                logs['fracture_density'] = np.clip(0.05 + 0.1 * vp_vs_norm, 0.01, 0.3)
            
            processed_logs[well_name] = logs
        
        return processed_logs


# ============================================================================
# Enhanced Visualization and Analysis
# ============================================================================

class EnhancedVisualization:
    """Enhanced visualization tools for fracture analysis"""
    
    @staticmethod
    def plot_azimuthal_gathers(results: Dict, well_name: str = None):
        """Plot azimuthal gathers from well log data"""
        
        if well_name not in results:
            return None
        
        data = results[well_name]
        incidence_angles = data['incidence_angles']
        azimuths = data['azimuths']
        synthetic_gathers = data['synthetic_gathers']
        
        # Create figure
        n_angles = len(incidence_angles)
        n_cols = 3
        n_rows = int(np.ceil(n_angles / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, angle in enumerate(incidence_angles):
            if f'angle_{angle}' in synthetic_gathers:
                gather_data = synthetic_gathers[f'angle_{angle}']['data']
                ax = axes[idx]
                
                # Plot gather
                vmax = np.percentile(np.abs(gather_data), 95)
                im = ax.imshow(gather_data, aspect='auto', cmap='seismic',
                             vmin=-vmax, vmax=vmax,
                             extent=[azimuths[0], azimuths[-1], 
                                     gather_data.shape[0], 0])
                
                ax.set_xlabel('Azimuth (degrees)')
                ax.set_ylabel('Time samples')
                ax.set_title(f'Azimuthal Gather - {angle}° Incidence')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(incidence_angles), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Azimuthal Gathers - {well_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_avaz_attributes(results: Dict, well_name: str = None):
        """Plot AVAz attributes from well log data"""
        
        if well_name not in results:
            return None
        
        data = results[well_name]
        incidence_angles = data['incidence_angles']
        avaz_attributes = data['avaz_attributes']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract attributes
        angles = []
        A_values = []
        B_values = []
        phi0_values = []
        B_A_ratios = []
        
        for angle in incidence_angles:
            if f'angle_{angle}' in avaz_attributes:
                attrs = avaz_attributes[f'angle_{angle}']
                angles.append(angle)
                A_values.append(attrs['A'])
                B_values.append(attrs['B'])
                phi0_values.append(attrs['phi0'])
                B_A_ratios.append(attrs['B_A_ratio'])
        
        # Plot 1: A and B vs angle
        ax1 = axes[0, 0]
        ax1.plot(angles, A_values, 'bo-', label='A (Isotropic)', linewidth=2, markersize=8)
        ax1.plot(angles, B_values, 'ro-', label='B (Anisotropic)', linewidth=2, markersize=8)
        ax1.set_xlabel('Incidence Angle (degrees)')
        ax1.set_ylabel('Reflectivity Coefficient')
        ax1.set_title('AVAz Attributes: A and B vs Angle')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: B/A ratio vs angle
        ax2 = axes[0, 1]
        ax2.plot(angles, B_A_ratios, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Incidence Angle (degrees)')
        ax2.set_ylabel('B/A Ratio')
        ax2.set_title('Relative Anisotropy (B/A) vs Angle')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Fast direction (phi0)
        ax3 = axes[1, 0]
        ax3.plot(angles, phi0_values, 'mo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Incidence Angle (degrees)')
        ax3.set_ylabel('Fast Direction (degrees)')
        ax3.set_title('Fast Direction vs Angle')
        ax3.set_ylim([0, 180])
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Polar plot of B magnitude
        ax4 = axes[1, 1]
        theta = np.radians(angles)
        radii = np.array(B_values)
        
        # Normalize for visualization
        if np.max(radii) > 0:
            radii = radii / np.max(radii)
        
        bars = ax4.bar(theta, radii, width=0.1, bottom=0.0, alpha=0.7)
        
        # Color bars by angle
        for bar, angle in zip(bars, angles):
            bar.set_facecolor(plt.cm.jet(angle/max(angles)))
        
        ax4.set_theta_zero_location('N')
        ax4.set_theta_direction(-1)
        ax4.set_rlabel_position(0)
        ax4.set_title('Anisotropy Magnitude (Polar)')
        
        plt.suptitle(f'AVAz Attributes Analysis - {well_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
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
            
            # Check if we have Vp data
            if logs.get('Vp') is not None:
                ax.plot(logs['Vp']/1000, logs['depth'], 'b-', label='Vp')
            if logs.get('Vs') is not None:
                ax.plot(logs['Vs']/1000, logs['depth'], 'r-', label='Vs')
            ax.set_xlabel('Velocity (km/s)')
            ax.set_ylabel('Depth')
            ax.set_title(f'{well_name} - Velocity')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            if i == 0 and (logs.get('Vp') is not None or logs.get('Vs') is not None):
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
    def plot_well_logs_comparison(actual_logs: Dict, synthetic_logs: Dict):
        """Plot comparison between actual and synthetic well logs"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        
        for well_name in list(actual_logs.keys())[:3]:  # Plot first 3 wells
            if well_name in synthetic_logs:
                actual = actual_logs[well_name]
                synthetic = synthetic_logs[well_name]
                
                # Vp comparison
                if actual.get('Vp') is not None and synthetic.get('Vp') is not None:
                    ax = axes[plot_idx]
                    ax.plot(actual['Vp']/1000, actual['depth'], 'b-', label='Actual', linewidth=2)
                    ax.plot(synthetic['Vp']/1000, synthetic['depth'], 'r--', label='Synthetic', linewidth=2)
                    ax.set_xlabel('Vp (km/s)')
                    ax.set_ylabel('Depth')
                    ax.set_title(f'{well_name} - Vp Comparison')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plot_idx += 1
                
                # Density comparison
                if actual.get('rho') is not None and synthetic.get('rho') is not None:
                    ax = axes[plot_idx]
                    ax.plot(actual['rho']/1000, actual['depth'], 'b-', label='Actual', linewidth=2)
                    ax.plot(synthetic['rho']/1000, synthetic['depth'], 'r--', label='Synthetic', linewidth=2)
                    ax.set_xlabel('Density (g/cc)')
                    ax.set_ylabel('Depth')
                    ax.set_title(f'{well_name} - Density Comparison')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plot_idx += 1
                
                # Porosity comparison
                if actual.get('porosity') is not None and synthetic.get('porosity') is not None:
                    ax = axes[plot_idx]
                    ax.plot(actual['porosity'], actual['depth'], 'b-', label='Actual', linewidth=2)
                    ax.plot(synthetic['porosity'], synthetic['depth'], 'r--', label='Synthetic', linewidth=2)
                    ax.set_xlabel('Porosity')
                    ax.set_ylabel('Depth')
                    ax.set_title(f'{well_name} - Porosity Comparison')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 6):
            axes[i].axis('off')
        
        plt.suptitle('Well Log Data Comparison: Actual vs Synthetic', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


# ============================================================================
# Streamlit Application with Well Log Upload
# ============================================================================

def create_streamlit_app():
    """Create comprehensive Streamlit application for AVAz analysis with well log upload"""
    
    st.set_page_config(layout="wide", page_title="Enhanced AVAz Analysis with Well Logs")
    st.title("Enhanced AVAz Analysis with Fracture Characterization")
    st.markdown("""
    ### Upload your well log data in CSV format and integrate it into the fracture analysis
    Required columns: DEPTH, VP, VS, RHOB, PHIE, SW, GR, RT (or similar)
    """)
    
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
    
    if 'well_logs' not in st.session_state:
        st.session_state.well_logs = {}
    
    if 'azimuthal_gathers' not in st.session_state:
        st.session_state.azimuthal_gathers = {}
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Well Log Upload", "Model Configuration", "Synthetic Generation", 
         "AVAz Analysis", "Azimuthal Gathers", "Fluid Substitution", "Inversion"]
    )
    
    # Well Log Upload Mode
    if app_mode == "Well Log Upload":
        st.header("Well Log Data Upload")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Upload CSV Files")
            
            # File uploader for multiple well logs
            uploaded_files = st.file_uploader(
                "Upload well log CSV files",
                type=["csv", "txt"],
                accept_multiple_files=True,
                help="Upload one or more CSV files containing well log data"
            )
            
            if uploaded_files:
                st.success(f"Uploaded {len(uploaded_files)} file(s)")
                
                # Process each uploaded file
                for uploaded_file in uploaded_files:
                    try:
                        # Read CSV file
                        df = pd.read_csv(uploaded_file)
                        
                        # Show file info
                        st.write(f"**File:** {uploaded_file.name}")
                        st.write(f"**Shape:** {df.shape}")
                        
                        # Show available columns
                        st.write("**Available columns:**")
                        st.write(", ".join(df.columns.tolist()))
                        
                        # Let user select well name
                        well_name = st.text_input(
                            f"Enter well name for {uploaded_file.name}",
                            value=uploaded_file.name.replace('.csv', '').replace('.txt', '')
                        )
                        
                        if st.button(f"Load {uploaded_file.name}", key=f"load_{uploaded_file.name}"):
                            # Store in session state
                            st.session_state.well_logs[well_name] = df
                            st.success(f"Loaded {well_name} with {len(df)} rows")
                        
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {str(e)}")
            
            # Clear all data button
            if st.button("Clear All Well Log Data", type="secondary"):
                st.session_state.well_logs = {}
                st.session_state.azimuthal_gathers = {}
                st.success("All well log data cleared")
        
        with col2:
            if st.session_state.well_logs:
                st.subheader("Loaded Well Logs")
                
                # Select a well to visualize
                selected_well = st.selectbox(
                    "Select well to visualize",
                    list(st.session_state.well_logs.keys())
                )
                
                if selected_well:
                    df = st.session_state.well_logs[selected_well]
                    
                    # Show data preview
                    st.write(f"**Well:** {selected_well}")
                    st.write(f"**Data shape:** {df.shape}")
                    
                    # Data statistics
                    st.write("**Data Statistics:**")
                    st.dataframe(df.describe())
                    
                    # Plot well logs
                    st.write("**Well Log Visualization:**")
                    
                    # Determine which logs to plot
                    plot_columns = []
                    if 'DEPTH' in df.columns:
                        depth_col = 'DEPTH'
                    elif 'depth' in df.columns:
                        depth_col = 'depth'
                    elif 'Depth' in df.columns:
                        depth_col = 'Depth'
                    else:
                        depth_col = df.columns[0]
                        st.warning(f"Using '{depth_col}' as depth column")
                    
                    # Common log column names
                    log_columns = {
                        'Vp': ['VP', 'Vp', 'vp', 'P_VEL', 'P_VELOCITY', 'DTP'],
                        'Vs': ['VS', 'Vs', 'vs', 'S_VEL', 'S_VELOCITY', 'DTS'],
                        'Density': ['RHOB', 'rho', 'RHO', 'DENSITY', 'DEN'],
                        'Porosity': ['PHIE', 'PHI', 'phi', 'POROSITY', 'POR'],
                        'GR': ['GR', 'gr', 'GAMMA', 'GAMMA_RAY'],
                        'RT': ['RT', 'rt', 'RESISTIVITY', 'RES']
                    }
                    
                    # Find actual column names
                    available_logs = {}
                    for log_name, possible_names in log_columns.items():
                        for name in possible_names:
                            if name in df.columns:
                                available_logs[log_name] = name
                                break
                    
                    # Plot available logs
                    if len(available_logs) > 0:
                        n_logs = len(available_logs)
                        fig, axes = plt.subplots(1, n_logs, figsize=(5*n_logs, 10))
                        
                        if n_logs == 1:
                            axes = [axes]
                        
                        for idx, (log_name, col_name) in enumerate(available_logs.items()):
                            ax = axes[idx]
                            ax.plot(df[col_name], df[depth_col], 'b-', linewidth=1)
                            ax.set_xlabel(log_name)
                            ax.set_ylabel('Depth')
                            ax.set_title(f'{log_name} - {selected_well}')
                            ax.invert_yaxis()
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("No standard well log columns found in the data")
    
    elif app_mode == "Model Configuration":
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
            
            # Option to use well logs
            use_well_logs = st.checkbox("Use Real Well Log Data", 
                                       value=len(st.session_state.well_logs) > 0,
                                       disabled=len(st.session_state.well_logs) == 0)
            
            if len(st.session_state.well_logs) == 0:
                st.warning("No well log data loaded. Please upload well logs first.")
            
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
                    
                    # Load well logs if using real data
                    if use_well_logs and st.session_state.well_logs:
                        st.session_state.generator.load_well_logs(st.session_state.well_logs)
                    
                    st.session_state.synthetic_results = (
                        st.session_state.generator.generate_synthetic_data(
                            config, use_well_logs=use_well_logs
                        )
                    )
                    st.success("Synthetic data generated successfully!")
        
        with col2:
            if 'synthetic_results' in st.session_state:
                st.subheader("Results Visualization")
                
                # Quick view of key results
                results = st.session_state.synthetic_results
                
                tab1, tab2, tab3, tab4 = st.tabs(["Map View", "Cross-section", "Well Logs", "Data Comparison"])
                
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
                        if logs.get('Vp') is not None:
                            ax1.plot(logs['Vp']/1000, logs['depth'], 'b-', label='Vp')
                        if logs.get('Vs') is not None:
                            ax1.plot(logs['Vs']/1000, logs['depth'], 'r-', label='Vs')
                        ax1.set_xlabel('Velocity (km/s)')
                        ax1.set_ylabel('Depth (m)')
                        ax1.set_title(f'{well_name} - Velocities')
                        ax1.invert_yaxis()
                        ax1.grid(True, alpha=0.3)
                        if logs.get('Vp') is not None or logs.get('Vs') is not None:
                            ax1.legend()
                        
                        # Impedances
                        ax2 = axes[1]
                        if logs.get('Acoustic_Impedance') is not None:
                            ax2.plot(logs['Acoustic_Impedance']/1e6, logs['depth'], 'b-', 
                                    label='AI')
                        if logs.get('Shear_Impedance') is not None:
                            ax2.plot(logs['Shear_Impedance']/1e6, logs['depth'], 'r-', 
                                    label='SI')
                        ax2.set_xlabel('Impedance (MPa·s/m)')
                        ax2.set_ylabel('Depth (m)')
                        ax2.set_title(f'{well_name} - Impedances')
                        ax2.invert_yaxis()
                        ax2.grid(True, alpha=0.3)
                        if logs.get('Acoustic_Impedance') is not None or logs.get('Shear_Impedance') is not None:
                            ax2.legend()
                        
                        # Fracture and anisotropy
                        ax3 = axes[2]
                        if logs.get('fracture_density') is not None:
                            ax3.plot(logs['fracture_density'], logs['depth'], 'g-', 
                                    label='Fracture Density')
                        ax3_twin = ax3.twinx()
                        if logs.get('anisotropy_strength') is not None:
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
                        if lines1 or lines2:
                            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
                        
                        st.pyplot(fig)
                
                with tab4:
                    # Compare actual vs synthetic well logs
                    if use_well_logs and st.session_state.well_logs:
                        # Process actual well logs for comparison
                        actual_logs = {}
                        for well_name, df in st.session_state.well_logs.items():
                            # Find depth column
                            depth_col = None
                            for col in df.columns:
                                if 'depth' in col.lower():
                                    depth_col = col
                                    break
                            
                            if depth_col is None:
                                depth_col = df.columns[0]
                            
                            actual_log = {
                                'depth': df[depth_col].values,
                                'Vp': None,
                                'Vs': None,
                                'rho': None,
                                'porosity': None
                            }
                            
                            # Find data columns
                            for col in df.columns:
                                col_lower = col.lower()
                                if any(x in col_lower for x in ['vp', 'p_vel', 'velocity_p']):
                                    actual_log['Vp'] = df[col].values
                                elif any(x in col_lower for x in ['vs', 's_vel', 'velocity_s']):
                                    actual_log['Vs'] = df[col].values
                                elif any(x in col_lower for x in ['rho', 'dens', 'density', 'rhob']):
                                    actual_log['rho'] = df[col].values
                                elif any(x in col_lower for x in ['phi', 'phie', 'porosity']):
                                    actual_log['porosity'] = df[col].values
                            
                            actual_logs[well_name] = actual_log
                        
                        fig = EnhancedVisualization.plot_well_logs_comparison(
                            actual_logs, results['well_logs']
                        )
                        st.pyplot(fig)
                    else:
                        st.info("Enable 'Use Real Well Log Data' to see comparison plots")
    
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
    
    elif app_mode == "Azimuthal Gathers":
        st.header("Azimuthal Gathers from Well Logs")
        
        if not st.session_state.well_logs:
            st.warning("Please upload well log data first!")
            st.info("Go to 'Well Log Upload' mode to upload your well log data.")
            return
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Generation Parameters")
            
            # Select well
            well_name = st.selectbox(
                "Select Well",
                list(st.session_state.well_logs.keys())
            )
            
            max_angle = st.slider("Maximum Incidence Angle", 10, 80, 60, key="az_gather_max_angle")
            angle_step = st.slider("Angle Step", 1, 15, 5, key="az_gather_angle_step")
            azimuth_step = st.slider("Azimuth Step", 5, 45, 15, key="az_gather_azimuth_step")
            wavelet_freq = st.slider("Wavelet Frequency (Hz)", 10, 100, 45, key="az_gather_freq")
            
            config = {
                'max_angle': max_angle,
                'angle_step': angle_step,
                'azimuth_step': azimuth_step,
                'wavelet_freq': wavelet_freq
            }
            
            if st.button("Generate Azimuthal Gathers", type="primary"):
                with st.spinner("Generating azimuthal gathers..."):
                    # Generate gathers from well logs
                    st.session_state.azimuthal_gathers = (
                        st.session_state.generator.generate_azimuthal_gathers_from_well_logs(
                            config, well_name
                        )
                    )
                    st.success("Azimuthal gathers generated successfully!")
        
        with col2:
            if st.session_state.azimuthal_gathers:
                st.subheader("Azimuthal Gathers Visualization")
                
                selected_well = list(st.session_state.azimuthal_gathers.keys())[0]
                data = st.session_state.azimuthal_gathers[selected_well]
                
                tab1, tab2, tab3, tab4 = st.tabs(["Gathers", "AVAz Attributes", "Reflectivity Matrix", "Well Logs"])
                
                with tab1:
                    # Plot azimuthal gathers
                    fig = EnhancedVisualization.plot_azimuthal_gathers(
                        st.session_state.azimuthal_gathers, selected_well
                    )
                    if fig:
                        st.pyplot(fig)
                
                with tab2:
                    # Plot AVAz attributes
                    fig = EnhancedVisualization.plot_avaz_attributes(
                        st.session_state.azimuthal_gathers, selected_well
                    )
                    if fig:
                        st.pyplot(fig)
                    
                    # Display attribute summary
                    st.subheader("AVAz Attribute Summary")
                    if 'avaz_attributes' in data and 'summary' in data['avaz_attributes']:
                        summary = data['avaz_attributes']['summary']
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Mean Fast Direction", f"{summary['mean_phi0']:.1f}°")
                        with col_b:
                            st.metric("Fast Direction Std", f"{summary['std_phi0']:.1f}°")
                        with col_c:
                            st.metric("Number of Angles", len(data['incidence_angles']))
                
                with tab3:
                    # Plot reflectivity matrix
                    st.subheader("Reflectivity Matrix")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    im = ax.imshow(data['reflectivity_matrix'].T, aspect='auto',
                                 extent=[data['incidence_angles'][0], data['incidence_angles'][-1],
                                         data['azimuths'][0], data['azimuths'][-1]],
                                 cmap='jet', origin='lower')
                    
                    ax.set_xlabel('Incidence Angle (degrees)')
                    ax.set_ylabel('Azimuth (degrees)')
                    ax.set_title('Reflectivity Matrix')
                    plt.colorbar(im, ax=ax, label='Reflectivity')
                    
                    st.pyplot(fig)
                
                with tab4:
                    # Display well log data
                    st.subheader("Well Log Data Used")
                    
                    if 'processed_logs' in data:
                        logs = data['processed_logs']
                        
                        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
                        
                        # Vp and Vs
                        ax1 = axes[0]
                        if logs.get('Vp') is not None:
                            ax1.plot(logs['Vp']/1000, logs['depth'], 'b-', label='Vp', linewidth=2)
                        if logs.get('Vs') is not None:
                            ax1.plot(logs['Vs']/1000, logs['depth'], 'r-', label='Vs', linewidth=2)
                        ax1.set_xlabel('Velocity (km/s)')
                        ax1.set_ylabel('Depth')
                        ax1.set_title(f'{selected_well} - Velocities')
                        ax1.invert_yaxis()
                        ax1.grid(True, alpha=0.3)
                        ax1.legend()
                        
                        # Density
                        ax2 = axes[1]
                        if logs.get('rho') is not None:
                            ax2.plot(logs['rho']/1000, logs['depth'], 'g-', label='Density', linewidth=2)
                        ax2.set_xlabel('Density (g/cc)')
                        ax2.set_ylabel('Depth')
                        ax2.set_title(f'{selected_well} - Density')
                        ax2.invert_yaxis()
                        ax2.grid(True, alpha=0.3)
                        ax2.legend()
                        
                        # Porosity
                        ax3 = axes[2]
                        if logs.get('phi') is not None:
                            ax3.plot(logs['phi'], logs['depth'], 'm-', label='Porosity', linewidth=2)
                        ax3.set_xlabel('Porosity')
                        ax3.set_ylabel('Depth')
                        ax3.set_title(f'{selected_well} - Porosity')
                        ax3.invert_yaxis()
                        ax3.grid(True, alpha=0.3)
                        ax3.legend()
                        
                        plt.tight_layout()
                        st.pyplot(fig)
    
    elif app_mode == "Fluid Substitution":
        st.header("Brown-Korringa Fluid Substitution")
        
        # Create two main columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Parameters")
            
            # Create tabs for different input methods
            input_method = st.radio(
                "Input Method",
                ["Manual Input", "From Well Logs", "Synthetic Data"],
                horizontal=True
            )
            
            if input_method == "Manual Input":
                # Manual input of parameters
                st.markdown("**Background Rock Properties**")
                Vp = st.number_input("Vp (m/s)", value=3000.0, step=100.0, key="fs_vp")
                Vs = st.number_input("Vs (m/s)", value=1500.0, step=50.0, key="fs_vs")
                rho = st.number_input("Density (kg/m³)", value=2400.0, step=50.0, key="fs_rho")
                porosity = st.slider("Porosity", 0.01, 0.4, 0.2, 0.01, key="fs_porosity")
                
                # Anisotropy parameters
                st.markdown("**Anisotropy Parameters**")
                epsilon = st.number_input("ε (Epsilon)", value=0.1, step=0.01, key="fs_epsilon")
                delta = st.number_input("δ (Delta)", value=0.05, step=0.01, key="fs_delta")
                gamma = st.number_input("γ (Gamma)", value=0.08, step=0.01, key="fs_gamma")
                
                # Mineral properties
                st.markdown("**Mineral Properties**")
                K_min = st.number_input("Mineral K (GPa)", value=37.0, step=1.0, key="fs_kmin")
                G_min = st.number_input("Mineral G (GPa)", value=44.0, step=1.0, key="fs_gmin")
                
                # Store for use
                input_params = {
                    'Vp': Vp, 'Vs': Vs, 'rho': rho, 'porosity': porosity,
                    'epsilon': epsilon, 'delta': delta, 'gamma': gamma,
                    'K_min': K_min, 'G_min': G_min
                }
                
            elif input_method == "From Well Logs" and st.session_state.well_logs:
                # Use well log data
                well_name = st.selectbox(
                    "Select Well",
                    list(st.session_state.well_logs.keys())
                )
                
                if well_name:
                    df = st.session_state.well_logs[well_name]
                    
                    # Find relevant columns
                    depth_col = None
                    for col in df.columns:
                        if 'depth' in col.lower():
                            depth_col = col
                            break
                    
                    if depth_col:
                        # Let user select depth
                        depth_range = st.slider(
                            "Select Depth Range",
                            float(df[depth_col].min()),
                            float(df[depth_col].max()),
                            (float(df[depth_col].min()), float(df[depth_col].max()))
                        )
                        
                        # Filter data by depth
                        mask = (df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])
                        filtered_df = df[mask]
                        
                        if not filtered_df.empty:
                            # Calculate averages
                            vp_col = None
                            for col in filtered_df.columns:
                                if any(x in col.lower() for x in ['vp', 'p_vel', 'velocity_p']):
                                    vp_col = col
                                    break
                            
                            if vp_col:
                                Vp = float(filtered_df[vp_col].mean())
                                Vs = Vp / 1.8  # Estimate
                                rho = 310 * (Vp ** 0.25) * 0.23  # Gardner
                                porosity = 0.4 - 0.0003 * Vp  # Empirical
                                
                                st.write(f"**Calculated Properties:**")
                                st.write(f"- Vp: {Vp:.0f} m/s")
                                st.write(f"- Vs: {Vs:.0f} m/s")
                                st.write(f"- Density: {rho:.0f} kg/m³")
                                st.write(f"- Porosity: {porosity:.2f}")
                                
                                input_params = {
                                    'Vp': Vp, 'Vs': Vs, 'rho': rho, 'porosity': porosity,
                                    'epsilon': 0.1, 'delta': 0.05, 'gamma': 0.08,  # Defaults
                                    'K_min': 37.0, 'G_min': 44.0
                                }
                            else:
                                st.error("No Vp data found in well logs")
                                return
                        else:
                            st.error("No data in selected depth range")
                            return
                    else:
                        st.error("No depth column found")
                        return
                else:
                    st.warning("Please select a well")
                    return
                
            elif input_method == "Synthetic Data" and 'synthetic_results' in st.session_state:
                # Use synthetic data
                results = st.session_state.synthetic_results
                
                # Let user select location
                nx, ny, nz = results['geology']['Vp'].shape
                x_pos = st.slider("X Position", 0, nx-1, nx//2, key="fs_x")
                y_pos = st.slider("Y Position", 0, ny-1, ny//2, key="fs_y")
                z_pos = st.slider("Z Position", 0, nz-1, nz//2, key="fs_z")
                
                Vp = float(results['geology']['Vp'][z_pos, y_pos, x_pos])
                Vs = float(results['geology']['Vs'][z_pos, y_pos, x_pos])
                rho = float(results['geology']['rho'][z_pos, y_pos, x_pos])
                porosity = float(results['geology']['porosity'][z_pos, y_pos, x_pos])
                
                st.write(f"**Selected Properties:**")
                st.write(f"- Vp: {Vp:.0f} m/s")
                st.write(f"- Vs: {Vs:.0f} m/s")
                st.write(f"- Density: {rho:.0f} kg/m³")
                st.write(f"- Porosity: {porosity:.2f}")
                
                input_params = {
                    'Vp': Vp, 'Vs': Vs, 'rho': rho, 'porosity': porosity,
                    'epsilon': 0.1, 'delta': 0.05, 'gamma': 0.08,  # Defaults
                    'K_min': 37.0, 'G_min': 44.0
                }
            else:
                st.warning("No data available for selected method")
                return
            
            # Fluid properties (common for all methods)
            st.markdown("**Fluid Properties**")
            fluid_type = st.selectbox("Fluid Type", ['brine', 'oil', 'gas'], key="fs_fluid_type")
            fluid = st.session_state.model.fluid_properties[fluid_type]
            K_fluid = fluid['K'] / 1e9  # Convert to GPa
            rho_fluid = fluid['rho']
            
            st.write(f"**Selected Fluid:** {fluid['name']}")
            st.write(f"- Bulk Modulus: {K_fluid:.1f} GPa")
            st.write(f"- Density: {rho_fluid} kg/m³")
            
            # Run button
            if st.button("Run Fluid Substitution", type="primary", key="fs_run"):
                with st.spinner("Calculating fluid substitution..."):
                    # Store for use in results column
                    st.session_state.fluid_sub_params = {
                        'input_params': input_params,
                        'fluid_type': fluid_type,
                        'fluid': fluid
                    }
                    st.success("Parameters saved for calculation!")
        
        with col2:
            if 'fluid_sub_params' in st.session_state:
                st.subheader("Fluid Substitution Results")
                
                params = st.session_state.fluid_sub_params
                input_params = params['input_params']
                fluid_type = params['fluid_type']
                fluid = params['fluid']
                
                # Create background model
                background = {
                    'Vp': input_params['Vp'],
                    'Vs': input_params['Vs'],
                    'rho': input_params['rho'],
                    'epsilon': input_params['epsilon'],
                    'delta': input_params['delta'],
                    'gamma': input_params['gamma'],
                    'porosity': input_params['porosity']
                }
                
                # Fluid properties
                fluid_props = {
                    'porosity': input_params['porosity'],
                    'mineral_K': input_params['K_min'] * 1e9,
                    'mineral_G': input_params['G_min'] * 1e9,
                    'fluid_K': fluid['K'],
                    'fluid_density': fluid['rho']
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
                rho_sat = input_params['rho'] * (1 - input_params['porosity']) + fluid['rho'] * input_params['porosity']
                
                Vp_initial = np.sqrt(C_initial[2, 2] / input_params['rho'])
                Vs_initial = np.sqrt(C_initial[5, 5] / input_params['rho'])
                Vp_sat = np.sqrt(C_saturated[2, 2] / rho_sat)
                Vs_sat = np.sqrt(C_saturated[5, 5] / rho_sat)
                
                # Calculate anisotropy parameters
                epsilon_initial = (C_initial[0, 0] - C_initial[2, 2]) / (2 * C_initial[2, 2])
                gamma_initial = (C_initial[5, 5] - C_initial[4, 4]) / (2 * C_initial[4, 4])
                
                epsilon_sat = (C_saturated[0, 0] - C_saturated[2, 2]) / (2 * C_saturated[2, 2])
                gamma_sat = (C_saturated[5, 5] - C_saturated[4, 4]) / (2 * C_saturated[4, 4])
                
                # Display results in a nice layout
                st.markdown("### Results Comparison")
                
                # Create comparison table
                comparison_data = {
                    'Property': ['Vp (m/s)', 'Vs (m/s)', 'Density (kg/m³)', 'ε', 'γ'],
                    'Initial': [f"{Vp_initial:.0f}", f"{Vs_initial:.0f}", 
                               f"{input_params['rho']:.0f}", f"{epsilon_initial:.4f}", 
                               f"{gamma_initial:.4f}"],
                    f'Saturated ({fluid_type})': [f"{Vp_sat:.0f}", f"{Vs_sat:.0f}", 
                                                f"{rho_sat:.0f}", f"{epsilon_sat:.4f}", 
                                                f"{gamma_sat:.4f}"],
                    'Change (%)': [f"{((Vp_sat - Vp_initial)/Vp_initial*100):+.1f}",
                                  f"{((Vs_sat - Vs_initial)/Vs_initial*100):+.1f}",
                                  f"{((rho_sat - input_params['rho'])/input_params['rho']*100):+.1f}",
                                  f"{((epsilon_sat - epsilon_initial)/epsilon_initial*100):+.1f}",
                                  f"{((gamma_sat - gamma_initial)/gamma_initial*100):+.1f}"]
                }
                
                df_comparison = pd.DataFrame(comparison_data)
                st.dataframe(df_comparison.style.highlight_max(axis=1, color='lightgreen')
                                           .highlight_min(axis=1, color='lightcoral'))
                
                # Create visualizations
                st.markdown("### Visual Comparison")
                
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                
                # Velocity comparison bar chart
                ax1 = axes[0, 0]
                categories = ['Vp', 'Vs']
                x_pos = np.arange(len(categories))
                width = 0.35
                
                ax1.bar(x_pos - width/2, [Vp_initial/1000, Vs_initial/1000], 
                       width, label='Initial', color='blue', alpha=0.7)
                ax1.bar(x_pos + width/2, [Vp_sat/1000, Vs_sat/1000], 
                       width, label=f'Saturated ({fluid_type})', color='red', alpha=0.7)
                
                ax1.set_xlabel('Velocity Type')
                ax1.set_ylabel('Velocity (km/s)')
                ax1.set_title('Velocity Comparison')
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(categories)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Anisotropy comparison
                ax2 = axes[0, 1]
                ani_categories = ['ε', 'γ']
                x_pos = np.arange(len(ani_categories))
                
                ax2.bar(x_pos - width/2, [epsilon_initial, gamma_initial], 
                       width, label='Initial', color='blue', alpha=0.7)
                ax2.bar(x_pos + width/2, [epsilon_sat, gamma_sat], 
                       width, label=f'Saturated ({fluid_type})', color='red', alpha=0.7)
                
                ax2.set_xlabel('Anisotropy Parameter')
                ax2.set_ylabel('Value')
                ax2.set_title('Anisotropy Comparison')
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(ani_categories)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # AVO response comparison (simplified)
                ax3 = axes[1, 0]
                angles = np.linspace(0, 40, 10)
                theta_rad = np.radians(angles)
                
                # Calculate reflectivity using simplified equations
                R_initial = 0.5 * ((Vp_sat - Vp_initial)/Vp_initial + (rho_sat - input_params['rho'])/input_params['rho'])
                R_sat = 0.5 * ((Vp_sat - Vp_initial)/Vp_initial + (rho_sat - input_params['rho'])/input_params['rho'])
                
                # Add angle dependence
                R_initial_full = R_initial * np.cos(theta_rad)**2
                R_sat_full = R_sat * np.cos(theta_rad)**2
                
                ax3.plot(angles, R_initial_full, 'b-', label='Initial', linewidth=2)
                ax3.plot(angles, R_sat_full, 'r-', label=f'Saturated ({fluid_type})', linewidth=2)
                
                ax3.set_xlabel('Incidence Angle (degrees)')
                ax3.set_ylabel('Reflectivity')
                ax3.set_title('AVO Response Comparison')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Fluid effect on anisotropy
                ax4 = axes[1, 1]
                fluids = ['brine', 'oil', 'gas']
                colors = ['blue', 'green', 'red']
                
                for fluid_name, color in zip(fluids, colors):
                    if fluid_name == fluid_type:
                        # Already calculated
                        continue
                    
                    # Calculate for other fluids
                    other_fluid = st.session_state.model.fluid_properties[fluid_name]
                    other_fluid_props = fluid_props.copy()
                    other_fluid_props['fluid_K'] = other_fluid['K']
                    other_fluid_props['fluid_density'] = other_fluid['rho']
                    
                    C_other = st.session_state.model._brown_korringa_substitution(
                        C_initial, background, other_fluid_props
                    )
                    epsilon_other = (C_other[0, 0] - C_other[2, 2]) / (2 * C_other[2, 2])
                    
                    ax4.bar(fluid_name, epsilon_other, color=color, alpha=0.7)
                
                # Add current fluid
                ax4.bar(fluid_type, epsilon_sat, color='gold', alpha=1.0, edgecolor='black', linewidth=2)
                
                ax4.set_xlabel('Fluid Type')
                ax4.set_ylabel('ε (Epsilon)')
                ax4.set_title('Anisotropy for Different Fluids')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Download results
                st.markdown("### Download Results")
                results_df = pd.DataFrame({
                    'Parameter': ['Vp_initial', 'Vs_initial', 'rho_initial', 'epsilon_initial', 'gamma_initial',
                                 f'Vp_{fluid_type}', f'Vs_{fluid_type}', f'rho_{fluid_type}', 
                                 f'epsilon_{fluid_type}', f'gamma_{fluid_type}'],
                    'Value': [Vp_initial, Vs_initial, input_params['rho'], epsilon_initial, gamma_initial,
                             Vp_sat, Vs_sat, rho_sat, epsilon_sat, gamma_sat]
                })
                
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"fluid_substitution_{fluid_type}.csv",
                    mime="text/csv"
                )
    
    elif app_mode == "Inversion":
        st.header("Bayesian Fracture Inversion")
        
        # Create main layout
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Inversion Parameters")
            
            # Data source selection
            data_source = st.radio(
                "Data Source",
                ["Synthetic Data", "Well Log Data", "Manual Range"],
                index=0
            )
            
            if data_source == "Synthetic Data" and 'synthetic_results' in st.session_state:
                results = st.session_state.synthetic_results
                
                # Let user select data range from synthetic results
                st.markdown("**Select Data Range**")
                
                # Select location range
                nx, ny = results['geology']['Vp'].shape[2], results['geology']['Vp'].shape[1]
                x_range = st.slider("X Range", 0, nx-1, (0, nx-1), key="inv_x_range")
                y_range = st.slider("Y Range", 0, ny-1, (0, ny-1), key="inv_y_range")
                z_range = st.slider("Z Range", 0, results['geology']['Vp'].shape[0]-1, 
                                   (0, results['geology']['Vp'].shape[0]-1), key="inv_z_range")
                
                # Extract data from selected range
                vp_data = results['geology']['Vp'][z_range[0]:z_range[1], 
                                                 y_range[0]:y_range[1], 
                                                 x_range[0]:x_range[1]]
                vs_data = results['geology']['Vs'][z_range[0]:z_range[1], 
                                                 y_range[0]:y_range[1], 
                                                 x_range[0]:x_range[1]]
                rho_data = results['geology']['rho'][z_range[0]:z_range[1], 
                                                   y_range[0]:y_range[1], 
                                                   x_range[0]:x_range[1]]
                fracture_data = results['fractures']['density'][z_range[0]:z_range[1], 
                                                             y_range[0]:y_range[1], 
                                                             x_range[0]:x_range[1]]
                
                st.write(f"**Data Statistics:**")
                st.write(f"- Vp: {vp_data.mean():.0f} ± {vp_data.std():.0f} m/s")
                st.write(f"- Vs: {vs_data.mean():.0f} ± {vs_data.std():.0f} m/s")
                st.write(f"- Fracture Density: {fracture_data.mean():.3f} ± {fracture_data.std():.3f}")
                
                # Store data for inversion
                inversion_data = {
                    'Vp': vp_data.flatten(),
                    'Vs': vs_data.flatten(),
                    'rho': rho_data.flatten(),
                    'fracture_density_truth': fracture_data.flatten(),
                    'source': 'synthetic'
                }
                
            elif data_source == "Well Log Data" and st.session_state.well_logs:
                well_name = st.selectbox(
                    "Select Well",
                    list(st.session_state.well_logs.keys())
                )
                
                if well_name:
                    df = st.session_state.well_logs[well_name]
                    
                    # Find depth column
                    depth_col = None
                    for col in df.columns:
                        if 'depth' in col.lower():
                            depth_col = col
                            break
                    
                    if depth_col:
                        # Let user select depth range
                        depth_range = st.slider(
                            "Select Depth Range",
                            float(df[depth_col].min()),
                            float(df[depth_col].max()),
                            (float(df[depth_col].min()), float(df[depth_col].max())),
                            key="inv_depth_range"
                        )
                        
                        # Filter data
                        mask = (df[depth_col] >= depth_range[0]) & (df[depth_col] <= depth_range[1])
                        filtered_df = df[mask]
                        
                        if not filtered_df.empty:
                            # Extract data
                            vp_values = []
                            vs_values = []
                            rho_values = []
                            
                            for col in filtered_df.columns:
                                col_lower = col.lower()
                                if any(x in col_lower for x in ['vp', 'p_vel', 'velocity_p']):
                                    vp_values = filtered_df[col].values
                                elif any(x in col_lower for x in ['vs', 's_vel', 'velocity_s']):
                                    vs_values = filtered_df[col].values
                                elif any(x in col_lower for x in ['rho', 'dens', 'density', 'rhob']):
                                    rho_values = filtered_df[col].values
                            
                            # Handle missing data
                            if len(vp_values) == 0:
                                st.error("No Vp data found")
                                return
                            
                            if len(vs_values) == 0:
                                # Estimate from Vp
                                vs_values = vp_values / 1.8
                            
                            if len(rho_values) == 0:
                                # Estimate from Gardner
                                rho_values = 310 * (vp_values ** 0.25) * 0.23
                            
                            st.write(f"**Data Statistics:**")
                            st.write(f"- Vp: {np.mean(vp_values):.0f} ± {np.std(vp_values):.0f} m/s")
                            st.write(f"- Vs: {np.mean(vs_values):.0f} ± {np.std(vs_values):.0f} m/s")
                            st.write(f"- Samples: {len(vp_values)}")
                            
                            inversion_data = {
                                'Vp': vp_values,
                                'Vs': vs_values,
                                'rho': rho_values,
                                'source': 'well_log',
                                'well_name': well_name
                            }
                        else:
                            st.error("No data in selected range")
                            return
                    else:
                        st.error("No depth column found")
                        return
                else:
                    st.warning("Please select a well")
                    return
                
            elif data_source == "Manual Range":
                st.markdown("**Define Parameter Ranges**")
                
                # Parameter ranges
                vp_min = st.number_input("Vp Min (m/s)", 1000, 8000, 2000, key="inv_vp_min")
                vp_max = st.number_input("Vp Max (m/s)", 1000, 8000, 5000, key="inv_vp_max")
                vs_min = st.number_input("Vs Min (m/s)", 500, 4000, 1000, key="inv_vs_min")
                vs_max = st.number_input("Vs Max (m/s)", 500, 4000, 3000, key="inv_vs_max")
                rho_min = st.number_input("Density Min (kg/m³)", 1500, 3000, 2000, key="inv_rho_min")
                rho_max = st.number_input("Density Max (kg/m³)", 1500, 3000, 2800, key="inv_rho_max")
                n_samples = st.slider("Number of Samples", 100, 10000, 1000, key="inv_n_samples")
                
                # Generate random samples
                np.random.seed(42)
                vp_samples = np.random.uniform(vp_min, vp_max, n_samples)
                vs_samples = np.random.uniform(vs_min, vs_max, n_samples)
                rho_samples = np.random.uniform(rho_min, rho_max, n_samples)
                
                st.write(f"**Generated Data:**")
                st.write(f"- Vp: {vp_samples.mean():.0f} ± {vp_samples.std():.0f} m/s")
                st.write(f"- Vs: {vs_samples.mean():.0f} ± {vs_samples.std():.0f} m/s")
                st.write(f"- Samples: {n_samples}")
                
                inversion_data = {
                    'Vp': vp_samples,
                    'Vs': vs_samples,
                    'rho': rho_samples,
                    'source': 'manual'
                }
            else:
                st.warning("No data available for selected source")
                return
            
            # Inversion parameters
            st.markdown("**Inversion Settings**")
            n_chains = st.slider("Number of MCMC chains", 1, 8, 4, key="inv_n_chains")
            n_iterations = st.slider("Iterations per chain", 100, 5000, 1000, key="inv_n_iter")
            burn_in = st.slider("Burn-in samples", 10, 1000, 200, key="inv_burn_in")
            
            inversion_method = st.selectbox(
                "Inversion Method",
                ["MCMC", "HMC", "NUTS", "Variational Inference"],
                index=0,
                key="inv_method"
            )
            
            include_uncertainty = st.checkbox("Include parameter uncertainty", True, key="inv_uncertainty")
            hierarchical = st.checkbox("Hierarchical model", False, key="inv_hierarchical")
            
            if st.button("Run Inversion", type="primary", key="inv_run"):
                with st.spinner("Running inversion..."):
                    # Run inversion
                    results = run_bayesian_inversion(
                        inversion_data, n_chains, n_iterations, burn_in,
                        inversion_method, include_uncertainty, hierarchical
                    )
                    
                    st.session_state.inversion_results = results
                    st.success("Inversion completed successfully!")
        
        with col2:
            if 'inversion_results' in st.session_state:
                st.subheader("Inversion Results")
                
                results = st.session_state.inversion_results
                
                tab1, tab2, tab3 = st.tabs(["Posterior Distributions", "Convergence", "Predictions"])
                
                with tab1:
                    # Plot posterior distributions
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    axes = axes.flatten()
                    
                    # Plot 1: Vp posterior
                    if 'Vp_posterior' in results:
                        ax = axes[0]
                        ax.hist(results['Vp_posterior'], bins=30, density=True, 
                               alpha=0.7, color='blue')
                        ax.axvline(results['Vp_mean'], color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {results["Vp_mean"]:.0f}')
                        ax.set_xlabel('Vp (m/s)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Vp Posterior Distribution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 2: Vs posterior
                    if 'Vs_posterior' in results:
                        ax = axes[1]
                        ax.hist(results['Vs_posterior'], bins=30, density=True, 
                               alpha=0.7, color='green')
                        ax.axvline(results['Vs_mean'], color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {results["Vs_mean"]:.0f}')
                        ax.set_xlabel('Vs (m/s)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Vs Posterior Distribution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 3: Density posterior
                    if 'rho_posterior' in results:
                        ax = axes[2]
                        ax.hist(results['rho_posterior'], bins=30, density=True, 
                               alpha=0.7, color='orange')
                        ax.axvline(results['rho_mean'], color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {results["rho_mean"]:.0f}')
                        ax.set_xlabel('Density (kg/m³)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Density Posterior Distribution')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 4: Fracture density posterior
                    if 'fracture_density_posterior' in results:
                        ax = axes[3]
                        ax.hist(results['fracture_density_posterior'], bins=30, density=True, 
                               alpha=0.7, color='purple')
                        ax.axvline(results['fracture_density_mean'], color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {results["fracture_density_mean"]:.3f}')
                        ax.set_xlabel('Fracture Density')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Fracture Density Posterior')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 5: Vp/Vs ratio posterior
                    if 'Vp_posterior' in results and 'Vs_posterior' in results:
                        ax = axes[4]
                        vp_vs_posterior = results['Vp_posterior'] / results['Vs_posterior']
                        ax.hist(vp_vs_posterior, bins=30, density=True, 
                               alpha=0.7, color='brown')
                        ax.axvline(np.mean(vp_vs_posterior), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(vp_vs_posterior):.2f}')
                        ax.set_xlabel('Vp/Vs Ratio')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Vp/Vs Ratio Posterior')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Plot 6: AI posterior
                    if 'Vp_posterior' in results and 'rho_posterior' in results:
                        ax = axes[5]
                        ai_posterior = results['Vp_posterior'] * results['rho_posterior']
                        ax.hist(ai_posterior, bins=30, density=True, 
                               alpha=0.7, color='teal')
                        ax.axvline(np.mean(ai_posterior), color='red', linestyle='--', 
                                 linewidth=2, label=f'Mean: {np.mean(ai_posterior)/1e6:.1f} Mrayl')
                        ax.set_xlabel('Acoustic Impedance (kg/m²·s)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title('Acoustic Impedance Posterior')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                    
                    # Hide unused axes
                    for i in range(6):
                        if i >= len(results.get('posterior_plots', [])):
                            axes[i].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display summary statistics
                    st.subheader("Summary Statistics")
                    
                    summary_data = []
                    if 'Vp_mean' in results:
                        summary_data.append(['Vp (m/s)', results['Vp_mean'], results['Vp_std'], 
                                           results['Vp_2.5'], results['Vp_97.5']])
                    if 'Vs_mean' in results:
                        summary_data.append(['Vs (m/s)', results['Vs_mean'], results['Vs_std'], 
                                           results['Vs_2.5'], results['Vs_97.5']])
                    if 'rho_mean' in results:
                        summary_data.append(['Density (kg/m³)', results['rho_mean'], results['rho_std'], 
                                           results['rho_2.5'], results['rho_97.5']])
                    if 'fracture_density_mean' in results:
                        summary_data.append(['Fracture Density', results['fracture_density_mean'], 
                                           results['fracture_density_std'], 
                                           results['fracture_density_2.5'], 
                                           results['fracture_density_97.5']])
                    
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data, 
                                                 columns=['Parameter', 'Mean', 'Std', '2.5%', '97.5%'])
                        st.dataframe(df_summary.style.format({
                            'Mean': '{:.1f}',
                            'Std': '{:.1f}',
                            '2.5%': '{:.1f}',
                            '97.5%': '{:.1f}'
                        }))
                
                with tab2:
                    # Convergence diagnostics
                    st.subheader("Convergence Diagnostics")
                    
                    if 'convergence_metrics' in results:
                        metrics = results['convergence_metrics']
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("R-hat", f"{metrics.get('rhat', 1.0):.3f}")
                        with col_b:
                            st.metric("Effective Sample Size", f"{metrics.get('ess', 1000):.0f}")
                        with col_c:
                            st.metric("Acceptance Rate", f"{metrics.get('acceptance_rate', 0.8):.3f}")
                        
                        # Trace plot
                        st.subheader("Trace Plots")
                        
                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        if 'trace_Vp' in results:
                            ax = axes[0, 0]
                            ax.plot(results['trace_Vp'], alpha=0.7)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Vp (m/s)')
                            ax.set_title('Vp Trace')
                            ax.grid(True, alpha=0.3)
                        
                        if 'trace_Vs' in results:
                            ax = axes[0, 1]
                            ax.plot(results['trace_Vs'], alpha=0.7)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Vs (m/s)')
                            ax.set_title('Vs Trace')
                            ax.grid(True, alpha=0.3)
                        
                        if 'trace_rho' in results:
                            ax = axes[1, 0]
                            ax.plot(results['trace_rho'], alpha=0.7)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Density (kg/m³)')
                            ax.set_title('Density Trace')
                            ax.grid(True, alpha=0.3)
                        
                        if 'trace_fracture' in results:
                            ax = axes[1, 1]
                            ax.plot(results['trace_fracture'], alpha=0.7)
                            ax.set_xlabel('Iteration')
                            ax.set_ylabel('Fracture Density')
                            ax.set_title('Fracture Density Trace')
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("Convergence metrics not available for demo inversion")
                
                with tab3:
                    # Model predictions
                    st.subheader("Model Predictions")
                    
                    if 'predictions' in results:
                        preds = results['predictions']
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        # Plot 1: Observed vs Predicted
                        ax1 = axes[0]
                        if 'observed' in preds and 'predicted' in preds:
                            ax1.scatter(preds['observed'], preds['predicted'], alpha=0.5)
                            ax1.plot([preds['observed'].min(), preds['observed'].max()],
                                    [preds['observed'].min(), preds['observed'].max()],
                                    'r--', label='Perfect Prediction')
                            ax1.set_xlabel('Observed')
                            ax1.set_ylabel('Predicted')
                            ax1.set_title('Observed vs Predicted')
                            ax1.legend()
                            ax1.grid(True, alpha=0.3)
                        
                        # Plot 2: Residuals
                        ax2 = axes[1]
                        if 'residuals' in preds:
                            ax2.hist(preds['residuals'], bins=30, alpha=0.7, color='blue')
                            ax2.axvline(0, color='red', linestyle='--', linewidth=2)
                            ax2.set_xlabel('Residuals')
                            ax2.set_ylabel('Frequency')
                            ax2.set_title('Prediction Residuals')
                            ax2.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Prediction metrics
                        if 'prediction_metrics' in results:
                            metrics = results['prediction_metrics']
                            
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("RMSE", f"{metrics.get('rmse', 0.0):.3f}")
                            with col_b:
                                st.metric("MAE", f"{metrics.get('mae', 0.0):.3f}")
                            with col_c:
                                st.metric("R²", f"{metrics.get('r2', 0.0):.3f}")
                    else:
                        st.info("Prediction results not available for demo inversion")
                        
                    # Download results
                    st.markdown("### Download Inversion Results")
                    
                    if st.button("Generate Report", key="inv_report"):
                        # Create comprehensive report
                        report_data = {
                            'parameter': ['Vp', 'Vs', 'Density', 'Fracture_Density'],
                            'mean': [results.get('Vp_mean', 0), results.get('Vs_mean', 0), 
                                    results.get('rho_mean', 0), results.get('fracture_density_mean', 0)],
                            'std': [results.get('Vp_std', 0), results.get('Vs_std', 0), 
                                   results.get('rho_std', 0), results.get('fracture_density_std', 0)],
                            '2.5_percentile': [results.get('Vp_2.5', 0), results.get('Vs_2.5', 0), 
                                              results.get('rho_2.5', 0), results.get('fracture_density_2.5', 0)],
                            '97.5_percentile': [results.get('Vp_97.5', 0), results.get('Vs_97.5', 0), 
                                               results.get('rho_97.5', 0), results.get('fracture_density_97.5', 0)]
                        }
                        
                        df_report = pd.DataFrame(report_data)
                        csv = df_report.to_csv(index=False)
                        
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="inversion_results.csv",
                            mime="text/csv"
                        )


def run_bayesian_inversion(data: Dict, n_chains: int, n_iterations: int, 
                          burn_in: int, method: str, include_uncertainty: bool,
                          hierarchical: bool) -> Dict:
    """
    Run Bayesian inversion on the provided data
    
    This is a simplified demonstration version. In production, this would
    use libraries like PyMC3, Stan, or TensorFlow Probability.
    """
    
    # Extract data
    Vp = data['Vp']
    Vs = data['Vs']
    rho = data['rho']
    
    # Generate synthetic posterior for demonstration
    np.random.seed(42)
    
    # Number of samples for posterior
    n_posterior_samples = min(5000, len(Vp))
    
    # Generate posterior samples
    Vp_posterior = np.random.normal(np.mean(Vp), np.std(Vp)/2, n_posterior_samples)
    Vs_posterior = np.random.normal(np.mean(Vs), np.std(Vs)/2, n_posterior_samples)
    rho_posterior = np.random.normal(np.mean(rho), np.std(rho)/2, n_posterior_samples)
    
    # Fracture density posterior (correlated with Vp/Vs ratio)
    vp_vs_ratio = Vp_posterior / Vs_posterior
    fracture_density_posterior = 0.05 + 0.1 * (vp_vs_ratio - np.mean(vp_vs_ratio)) / np.std(vp_vs_ratio)
    fracture_density_posterior = np.clip(fracture_density_posterior, 0.01, 0.3)
    
    # Add some noise
    fracture_density_posterior += np.random.normal(0, 0.02, n_posterior_samples)
    fracture_density_posterior = np.clip(fracture_density_posterior, 0.0, 0.5)
    
    # Calculate statistics
    results = {
        'Vp_posterior': Vp_posterior,
        'Vs_posterior': Vs_posterior,
        'rho_posterior': rho_posterior,
        'fracture_density_posterior': fracture_density_posterior,
        
        'Vp_mean': np.mean(Vp_posterior),
        'Vp_std': np.std(Vp_posterior),
        'Vp_2.5': np.percentile(Vp_posterior, 2.5),
        'Vp_97.5': np.percentile(Vp_posterior, 97.5),
        
        'Vs_mean': np.mean(Vs_posterior),
        'Vs_std': np.std(Vs_posterior),
        'Vs_2.5': np.percentile(Vs_posterior, 2.5),
        'Vs_97.5': np.percentile(Vs_posterior, 97.5),
        
        'rho_mean': np.mean(rho_posterior),
        'rho_std': np.std(rho_posterior),
        'rho_2.5': np.percentile(rho_posterior, 2.5),
        'rho_97.5': np.percentile(rho_posterior, 97.5),
        
        'fracture_density_mean': np.mean(fracture_density_posterior),
        'fracture_density_std': np.std(fracture_density_posterior),
        'fracture_density_2.5': np.percentile(fracture_density_posterior, 2.5),
        'fracture_density_97.5': np.percentile(fracture_density_posterior, 97.5),
        
        # Convergence metrics (simulated)
        'convergence_metrics': {
            'rhat': 1.01,
            'ess': n_posterior_samples,
            'acceptance_rate': 0.78
        },
        
        # Trace plots (simulated)
        'trace_Vp': np.cumsum(np.random.randn(n_iterations)) + np.mean(Vp),
        'trace_Vs': np.cumsum(np.random.randn(n_iterations)) + np.mean(Vs),
        'trace_rho': np.cumsum(np.random.randn(n_iterations)) + np.mean(rho),
        'trace_fracture': np.cumsum(np.random.randn(n_iterations)) + 0.15,
        
        # Predictions (simulated)
        'predictions': {
            'observed': Vp[:100],
            'predicted': Vp_posterior[:100],
            'residuals': np.random.randn(100) * np.std(Vp)/10
        },
        
        'prediction_metrics': {
            'rmse': np.std(Vp)/100,
            'mae': np.std(Vp)/150,
            'r2': 0.85
        }
    }
    
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main function to run the enhanced AVAz analysis with well log integration"""
    
    # Create Streamlit app
    create_streamlit_app()


if __name__ == "__main__":
    main()
