## Library of functions to run for APCEMM post processing

# Imports
import numpy as np
import os as os
from glob import glob
import netCDF4 as nc
import pandas as pd
import datetime as dt


## Useful meteorological functions

# Liquid water saturation pressure
def Psat_h2ol ( T ):
    return 100 * np.exp( -6096.9385/T + 16.635794 - 0.02711193*T  + 1.673952E-5*T**2 + 2.433502  *np.log(T))

# Solid water saturation pressure
def Psat_h2os ( T ):
    return 100 * np.exp( -6024.5282/T + 24.7219   + 0.010613868*T - 1.319883E-5*T**2 - 0.49382577*np.log(T));

# RHi from specific humidity and temperature
def QT2RHi( QV, TEMP, P ): # QV = kg/kg; T = K; P = hPa
    QVsat = 0.622 * Psat_h2os(TEMP) / ( P*100 )
    RHi = 100 * QV / QVsat
    return RHi

# RHi from concentration [molec/cm3]
def C2RHi( C, TEMP ):
    kB = 1.380649E-23
    RHi = 100 * C * kB * TEMP / Psat_h2os( TEMP ) * 1.0E+6
    return RHi

# Convert RHi to concentration
def RHi2C( RHi, T ):
    H2O = RHi/100 * Psat_h2os( T ) / ( 1.38E-23 * T * 1E6 )
    return H2O#[molecules/cm3]

# RHw to RHi
def RHw2RHi( RH, T ):
    RHi = RH * Psat_h2ol(T) / Psat_h2os(T)
    return RHi

# RHi to RHw
def RHi2RHw( RHi, T):
    RH = RHi * Psat_h2os(T) / Psat_h2ol(T)
    return RH

# SAC formulation
def SAC_general( RH, T, P, EIH2O=1.23, cp=1004, LHV=43.13, eta0=0.4, RHic=100, printout=0 ):
    # For given scalar or vector values of RHw, T and P, does a contrail form?
    
    # Calculate SAC constraints
    G = ( EIH2O*cp*P*100 ) / ( 0.622*LHV*1E6*(1-eta0) )
    Tc = 226.69 + 9.43*np.log(G-0.053) + 0.72*(np.log(G-0.053))**2
    RHc = 100 * ( G*(T-Tc) + Psat_h2ol( Tc ) ) / Psat_h2ol( T )
    
    # Calculate RHi
    RHi = RHw2RHi( RH, T )
    
    # Identify if pass
    SACpass = ((T<Tc) & (RH>RHc) & (RHi>RHic)) * 1
    
    return SACpass

# Excess H2O concentration above saturation
def excessH2Oconc( H2O, T ):
    N_A = 6.02214076E23
    R = 8.314
    psat_ice = Psat_h2os( T )
    H2O_ice = psat_ice*N_A/(R*T) * 1E-6
    excH2O = H2O - H2O_ice
    return excH2O



## APCEMM output processing scripts

# Open the netcdf file
def getAPCEMM( file ):
    # Pointer to the netcdf files
    try:
        nc_APCEMM = nc.Dataset(file, 'r', format='NETCDF4_CLASSIC')
    except:
        nc_APCEMM = 'null'
    return nc_APCEMM

# Get the time based on filename
def getAPCEMM_time( file ):
    # Extract hh:mm from filename
    filename = os.path.basename( file )
    hh = filename[-7:-5]
    mm = filename[-5:-3]
    return hh, mm

# Extract basic grid information
def getAPCEMM_grid( nc_APCEMM ):
    
    # Extract basic grid information
    X = nc_APCEMM['x'][:]
    Y = nc_APCEMM['y'][:].flatten()
    
    # Calculate grid cell values
    dX = X[2]-X[1]
    dY = Y[2]-Y[1]
    areaCell = dX*dY
    
    return X, Y, areaCell

# Extract background meteorology
def getAPCEMM_met( nc_APCEMM ):
    
    # Extract the T and water concentration
    T = nc_APCEMM['Temperature'][:]
    H2O = nc_APCEMM['H2O'][:]
    
    # Calculate RHi
    RHi = C2RHi( H2O, T )
    
    return T, RHi

# Extract 2D distribution of mass, particle number and radius
def getAPCEMM_2Ddist( nc_APCEMM ):
    
    # Extract information from netcdf files
    icenumber_vol = nc_APCEMM['Ice aerosol particle number'][:]
    icemass_vol = nc_APCEMM['Ice aerosol volume'][:] * 9.167E2
    effradius = nc_APCEMM['Effective radius'][:]
    
    return icenumber_vol, icemass_vol, effradius

def getAPCEMM_iceedge( icemass_vol, X, Y ):
    
    # Check left and lower domain for ice mass
    left_idx = 0
    lower_idx = len(Y)
    
    # Edge sizes
    dX = X[1] - X[0]
    dY = Y[1] - Y[0]
    areaCell = dX*dY
    
    # Sum icemass_vol
    icemass_miss = np.sum(icemass_vol[:,left_idx])*dX + np.sum(icemass_vol[lower_idx,:])*dY - \
                    icemass_vol[lower_idx,left_idx]*areaCell
    icemass_miss = icemass_miss * 1E6 * areaCell
    
    return icemass_miss


# Check no crystals falling out of domain boundaries
def checkAPCEMM_bounds( icenumber_vol, X, Y, Xcheck_left=5.0E3, Xcheck_right=5.0E3, Ycheck_up=400, Ycheck_down=200, num_lim=1E-5 ):
    
    # Identify grid cells to check for each boundary
    left_idx = np.argmin( np.abs( X - ( X[0] + Xcheck_left ) ) )
    right_idx = np.argmin( np.abs( X - ( X[-1] - Xcheck_right ) ) )
    down_idx = np.argmin( np.abs( Y - ( Y[0] + Ycheck_down ) ) ) - 1
    up_idx = np.argmin( np.abs( Y - ( Y[-1] - Ycheck_up ) ) ) + 1
    
    # Define output variable as 0 (not exceeding bounds) to begin with
    exceed = 0
    
    # dX, dY
    dX = X[2] - X[1]
    dY = Y[2] - Y[1]
    
    # Get totals
    total_left  = np.sum( icenumber_vol[:,0:left_idx]*dX )
    total_lower = np.sum( icenumber_vol[down_idx:,:]*dY )
    
    print(total_left, total_lower)
    
    # Check left edge
    if total_left >= num_lim:
        print('Left edge')
        exceed = 1
    
    # Check bottom edge
    if total_lower >= num_lim:
        print('Bottom edge')
        exceed = 1
    
    return exceed

# Calculate extinction coefficient
def calc_extinction( rE, IWC, rE_lim=1.00E-15 ):
    # Inputs:
    # - rE = effective radius [m]
    # - IWC = ice water content [kg/m3]
    
    # Define a, b parameters
    a = 3.448
    b = 2.431E-3
    
    # Calculate extinction
    chi = IWC * ( a + b/rE ) * ( rE > rE_lim ) + 0 * ( rE <= rE_lim )
    
    return chi

# Calculate predominant optical depth
def calc_preod( nc_APCEMM, X ):
    # Inputs:
    # - nc_APCEMM = pointer to APCEMM netcdf file
    # - X = X direction
    
    # Extract vertically intergrated OD
    OD_vertint = nc_APCEMM['Vertical optical depth'][:]
    
    # Calculate bin width in X direction
    dX = X[2] - X[1]
    
    # Use to calculate predominant OD
    preOD = np.sum(OD_vertint**2)*dX / (np.sum(OD_vertint)*dX)
    
    return preOD

# Calculate optical depth times width
def calc_odwidth( nc_APCEMM, X ):
    # Inputs:
    # - nc_APCEMM = pointer to APCEMM netcdf file
    # - X = X direction
    
    # Extract vertically intergrated OD
    OD_vertint = nc_APCEMM['Vertical optical depth'][:]
    
    # Calculate bin width in X direction
    dX = X[2] - X[1]
    
    # Use to calculate predominant OD
    odwidth = np.sum(OD_vertint)*dX
    
    return odwidth

# Calculate total from 2D distribution [assumed /cm3]
def getAPCEMM_tot( var, areaCell, const=1 ):
    
    var_tot = np.sum( var ) * areaCell * 1E+6 * const
    
    return var_tot

# Calculate total from 2D distribution [assumed /cm3]
def getAPCEMM_IWC( icemass_concentration, areaCell, const=1 ):
    
    var_tot = np.sum( var ) * areaCell * 1E+6 * const
    
    return var_tot

# Get upper and lower parts according 1% and 99% points
def getAPCEMM_upperlower( var_hint, Y ):
    
    # Total integral
    dY = Y[2] - Y[1]
    var_tot = np.sum( var_hint ) * dY
    var_cum = np.cumsum( var_hint ) * dY
    
    # Loop over until find lower portion
    for idx,val in enumerate(Y):
        var_tmp = var_cum[ idx ]
        if var_tmp/var_tot >= 0.001:
            var_lower = var_tmp/var_tot
            idx_lower = idx
            if idx > 0:
                idx_lower = idx - 1
            break
    
    # Loop over until find upper portion
    # for idx,val in enumerate(Y):
    Y2 = Y.copy()
    for idx,val in reversed( list( enumerate(Y2) ) ):
        var_tmp = var_cum[ idx ]
        if var_tmp/var_tot <= 0.999:
            var_upper = var_tmp/var_tot
            idx_upper = idx
            if idx < len(Y) - 1:
                idx_upper = idx+1
            break
    
    return Y[idx_lower], Y[idx_upper]

# Get upper and lower parts according to a defined limit
def getAPCEMM_upperlower_lim( var, Y, lim ):
    
    # Check some values above limit
    if np.all( var<lim ):
        return np.nan, np.nan
        
    # Find first and last time value is greater than limit
    idx_lower = np.argmax( var>lim ) - 1
    idx_upper = len(var) - np.argmax( var[::-1]>lim ) - 1
    
    return idx_lower, idx_upper



# Calculate horizontal integral
def hint( var, grid_size ):
    var_hint = np.sum(var,axis=1) * grid_size
    return var_hint

# Calculate vertical integral
def vint( var, grid_size ):
    var_vint = np.sum(var,axis=0) * grid_size
    return var_vint

# Get total ice mass and particle number that leave domain
def getAPCEMM_leftdomain( nc_APCEMM ):
    
    # Get information from APCEMM files
    icenumber_left = nc_APCEMM['Particle exits'][:]
    icemass_left = nc_APCEMM['Ice mass exits'][:]
    
    return icenumber_left, icemass_left

def checkavail( nc_APCEMM ):
    # Returns True if all desired variables found
    # Return False if any desired variables NOT found
    
    # Desired variables
    desired_variables = [ 'x', 'y', 'r', 'r_e', 'Pressure', 'Temperature',
                          'H2O', 'Ice aerosol particle number', 'Ice aerosol volume', 'Effective radius',
                          'Horizontal optical depth', 'Vertical optical depth' ]
    
    # Loop over
    found_var = np.zeros( len(desired_variables) )
    for ii, var in enumerate(desired_variables):
        # Check if desired variable in APCEMM netcdf file
        if var in nc_apcemm.variables:
            found_var[ ii ] = 1
        else:
            print( 'Not found %s' %(var) )
    
    # Return accordingly
    if np.sum(found_var) == len(found_var):
        return True
    else:
        return False






