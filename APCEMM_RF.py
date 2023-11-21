#!/usr/bin/env python3
import pandas as pd
import numpy as np
import netCDF4 as nc
import scipy.interpolate as sipl
#import scipy.integrate as sint
import apcemm_scripts as aps
import importlib
import os
from shutil import copyfile
import subprocess
import datetime as dt
import pysolar.solar as pys
from glob import glob
import numbers
import matplotlib.dates as mdates
import xarray as xr
import stat

# How to run
nproc = 1 # If nproc = 1, run in serial mode

###################################
# Functions
###################################

# Return grid dimensions after summing approach of APCEMM columns
def ncols_apcemm_colsums( ncol, approach=0, number2sum=16 ):
    
    if approach == 0:
        ncolsum = int( np.floor( ncol/number2sum ) )
    elif approach == 1:
        ncolsum = number2sum
    elif approach == 2:
        print('not ready yet')
    else:
        print('Approach not available... treat all columns separately')
        ncolsum = ncol
    
    return ncolsum

# Summing approach across columns of APCEMM
def apcemm_colsums( icemass_vol, icenumber_vol, effradius, X, approach=0, number2sum=16 ):
    # icemass_vol: [g/m^3]
    # icenumber_vol: [particles/m^3]
    # effradius: [m]
    # approach options:
    #   - 0 = group N columns together
    #   - 1 = group into N groups
    #   - 2 = group only where 99% of mass/number lies
    
    nrow, ncol = np.shape( icenumber_vol )
    width = X[2] - X[1]
    
    if approach == 0:
        # Group number2sum columns together
        
        # Calculate the number of columns after summing
        ncolsum = int( np.ceil( ncol/number2sum ) )
        # Pad with zeros if the division isn't neat
        if ncol%number2sum != 0:
            # Pad with zeros
            n_pad = ncolsum * number2sum
            icemass_pad = np.zeros((nrow,n_pad))
            icenum_pad  = np.zeros((nrow,n_pad))
            reff_pad    = np.zeros((nrow,n_pad))
            icemass_pad[:,:ncol] = icemass_vol[...]
            icenum_pad[:,:ncol]  = icenumber_vol[...]
            reff_pad[:,:ncol]    = effradius[...]
        else:
            icemass_pad = icemass_vol
            icenum_pad  = icenumber_vol
            reff_pad    = effradius
        
        # Zero out variables
        X_sum = np.zeros( (ncolsum) )
        width_sum = np.zeros( (ncolsum) )
        icemass_vol_sum = np.zeros( (nrow, ncolsum) )
        icenumber_vol_sum = np.zeros( (nrow, ncolsum) )
        effradius_sum = np.zeros( (nrow, ncolsum) )
        
        # Loop over each output, summed column
        jvalues = np.arange( 0, ncolsum, 1 ).astype(int)
        for idx,j in enumerate(jvalues):
            X_sum[idx] = np.mean( X[number2sum*j:number2sum*j+(number2sum-1)] )
            width_sum[idx] = width*( number2sum )
            icemass_vol_sum[:,idx] = np.sum( icemass_vol[ :, number2sum*j:number2sum*j+(number2sum-1) ], \
                                             axis=1 )/(number2sum-1)
            icenumber_vol_sum[:,idx] = np.sum( icenumber_vol[ :, number2sum*j:number2sum*j+(number2sum-1) ], \
                                               axis=1 )/(number2sum-1)
            effradius_sum[:,idx] = np.sum( icenumber_vol[ :, number2sum*j:number2sum*j+(number2sum-1) ]*\
                                           effradius[ :, number2sum*j:number2sum*j+(number2sum-1) ], \
                                           axis=1 )/icenumber_vol_sum[:,idx]/(number2sum-1)
        
    elif approach == 1:
        # Groups N columns such that you have number2sum columns
        
        # Zero out variables
        X_sum = np.zeros( (number2sum) )
        icemass_vol_sum = np.zeros( (nrow, number2sum) )
        icenumber_vol_sum = np.zeros( (nrow, number2sum) )
        effradius_sum = np.zeros( (nrow, number2sum) )
        width_sum = np.zeros( (number2sum) )
        
        # Loop over each summed column
        if ncol%number2sum == 0:
            ncols_per_summed = int( ncol/number2sum )
        else:
            raise ValueError('Cannot consistently group columns')
        
        for j in np.arange( 0, number2sum, 1 ).astype(int):
            X_sum[j] = np.mean( X[ ncols_per_summed*j:ncols_per_summed*j+(ncols_per_summed-1)] )
            icemass_vol_sum[:,j] = np.sum( icemass_vol[ :, ncols_per_summed*j:ncols_per_summed*j+(ncols_per_summed-1)], \
                                           axis=1 )/(ncols_per_summed-1)
            icenumber_vol_sum[:,j] = np.sum( icenumber_vol[ :, ncols_per_summed*j:ncols_per_summed*j+(ncols_per_summed-1)], \
                                             axis=1 )/(ncols_per_summed-1)
            effradius_sum[:,j] = np.sum( icenumber_vol[ :, ncols_per_summed*j:ncols_per_summed*j+(ncols_per_summed-1)]*\
                                         effradius[ :, ncols_per_summed*j:ncols_per_summed*j+(ncols_per_summed-1)], \
                                         axis=1 )/icenumber_vol_sum[:,j]/(ncols_per_summed-1)
            width_sum[j] = width*ncols_per_summed
        
    elif approach == 2:
        # Group number2sum columns (or less) together in region where 99% of contrail lies
        print('not ready yet')
        
    
    else:
        print( 'Approach not available... treat all columns separately' )
        
        # Return input values...
        icemass_vol_sum = icemass_vol
        icenumber_vol_sum = icenumber_vol
        effradius_sum = effradius
        X_sum = X
        width_sum = np.ones( ncol ) * width
    
    return icemass_vol_sum, icenumber_vol_sum, effradius_sum, X_sum, width_sum

def calc_cloud_paths_era5(cldfr, ql, qi, qr, qs, qv, phi, p_atm, T_atm):
    # Calculate cloud liquid and ice water paths given ERA5 input data
    # ERA5 variable names given in square brackets
    # cldfr   3D cloud fraction (0 to 1) [cc]
    # ql      Specific cloud liquid water content (kg/kg) [clwc]
    # qi      Specific cloud ice water content (kg/kg) [ciwc]
    # qr      Specific rain water content (kg/kg) [crwc]
    # qs      Specific snow water content (kg/kg) [cswc]
    # qv      Specific humidity (kg/kg) [q]
    # phi     Geopotential height (m2/s2) [z]
    # T_atm   Temperature (K) [t]
    # p_atm   Pressure at level centers (Pa) [level]
    
    rhoice = 0.9167 # g/cm3
    rholiq = 1.0000 # g/cm3
    rei_def = 24.8 # microns
    rel_def = 14.2 # microns
    
    Rd = 287.0597 # gas constant for dry air [J/kg/K]
    Rv = 461.5250 # gas constant for water vapor [J/kg/K]
    eps_star = Rv/Rd - 1

    g0 = 9.80665 # standard gravity, m/s^2
    Re = 6.3781e6 # average Earth radius, m
    
    height = phi * Re / (g0 * Re - phi) # m
    dz = np.diff(height)
    dz = np.append(dz, (dz[-1] - dz[-2] )/ (height[-2] - height[-3]) * (height[-1] - height[-2]) + dz[-1])

    Tv = T_atm*(1 + eps_star*qv - ql - qi - qr - qs)
    rho_dry = p_atm/Rd/Tv
    rho_moist = rho_dry/(1 - qv - ql - qi - qr - qs)

    cliqwp = ql * rho_moist * dz * 1e3 # g/m2
    cicewp = qi * rho_moist * dz * 1e3 # g/m2
    return cliqwp, cicewp

def calc_cloud_paths_merra2(cldfr, taucli, tauclw, cicewp, cliqwp):
    # cldfr   3D cloud fraction (0 to 1) [CLOUD]
    # taucli  Ice cloud optical depth (unitless) [TAUCLI]
    # tauclw  Water cloud optical depth (unitless) [TAUCLW]
    cicewp = 0.667*taucli*rhoice*rei_def
    cliqwp = 0.667*tauclw*rholiq*rel_def
    return cliqwp, cicewp

# Calculate RF for an APCEMM file
def APCEMM2RRTM_V2( apcemm_data_file,z_flight,
                    flight_datetime, number2sum, approach,
                    altitude_edges, fn_z_to_p,
                    temperature, relative_humidity,
                    ref_dir, min_icemass=1.0e-5, verbose=False,
                    emissivity=None,albnirdf=None,albnirdr=None,
                    albvisdf=None,albvisdr=None,sza=None,
                    cldfr=None,clwp=None,ciwp=None):
    # apcemm_data_file                Location of input ts_aerosol_etc file
    # z_flight                        Altitude at which the contrail is initiated (m)
    # flight_datetime                 Date of the flight (datetime object)
    # number2sum                      Parameter controlling how APCEMM columns are collapsed (see approach)
    # approach                        0: Sum together number2sum of columns
    #                                 1: Sum together columns into number2Performn only number2sum calculations
    # altitude_edges                  Layer edges of the met data (m)
    # fn_z_to_p                       A function to convert altitude in m to pressure in Pa
    # temperature                     Temperature in the met data (K)
    # relative_humidity               RH in the met data (0-1, or %?)
    # ref_dir                         Directory containing template RRTM input files
    # min_icemass                     Minimum ice mass in kg/m for processing [default: 1.0e-5 kg/m]
    # verbose                         Print output during processing [default: False]
    # emissivity                      Surface longwave emissivity (0-1)
    # albnirdf                        Surface albedo, near-IR diffuse (0-1)
    # albnirdr                        Surface albedo, near-IR direct (0-1)
    # albvisdf                        Surface albedo, visible diffuse (0-1)
    # albvisdr                        Surface albedo, visible direct (0-1)
    # sza                             Solar zenith angle (degrees)
    # cldfr                           Cloud fraction (0-1, unitless)
    # clwp                            Cloud liquid water path (g/m2)
    # ciwp                            Cloud ice water path (g/m2)
 
    # First: define a standard vertical grid onto which everything else will be mapped
    # This is currently assumed to match the vertical grid used by the met data
    altitude = (altitude_edges[1:] + altitude_edges[:-1])/2.0
    n_lev_met = altitude.size

    # Now, set default values
    if emissivity is None:
        emissivity_val = 0.1
    else:
        emissivity_val = emissivity
    if albnirdf is None:
        albnirdf = 0.30
    if albnirdr is None:
        albnirdr = 0.30
    if albvisdf is None:
        albvisdf = 0.30
    if albvisdr is None:
        albvisdr = 0.30
    if sza is None:
        sza = 30.0
    if clwp is None:
        clwp = np.zeros(n_lev_met)
    if ciwp is None:
        ciwp = np.zeros(n_lev_met)
    if cldfr is None:
        cldfr = np.zeros(n_lev_met)

    # Altitude in meters, pressure in Pa
    # Get the associated pressures and pressure edges (ISA)
    pressure_edges = fn_z_to_p(altitude_edges)
    pressure       = fn_z_to_p(altitude)

    # Retrieve data from the APCEMM simulation
    apcemm_folder = os.path.dirname( apcemm_data_file )
    apcemm_folder_rrtm = os.path.join(apcemm_folder,'rrtm')
    if not os.path.exists(apcemm_folder_rrtm):
        os.makedirs(apcemm_folder_rrtm)

    # Get apcemm data
    nc_apcemm = nc.Dataset( apcemm_data_file, 'r' )
    hh, mm = aps.getAPCEMM_time( apcemm_data_file )
    time = int(hh) + int(mm)/60
    if verbose:
         print( '    %s:%s' %( hh, mm ) )
    X, Y, areaCell = aps.getAPCEMM_grid( nc_apcemm )
    dY = Y[2] - Y[1]
    icenumber_vol, icemass_vol, effradius = aps.getAPCEMM_2Ddist( nc_apcemm )
    icenumber_vol *= 1.0E+06
    icemass_vol *= 1.0E+06 * 1.0E+03 # Converts from kg/cm3 to g/m^3
    
    # Variable size
    nrow, ncol = icenumber_vol.shape
    
    # Sum together multiple columns
    icemass_vol_sum, icenumber_vol_sum, effradius_sum, X_sum, width_sum = \
                apcemm_colsums( icemass_vol, icenumber_vol, \
                                effradius, X, approach=approach, \
                                number2sum=number2sum )
    
    # Convert APCEMM Y values to altitude and determine where those fall on the existing grid
    altitude_apcemm = Y.data + z_flight

    # Build interpolant which can translate an altitude to a pressure
    fn_z_to_p = sipl.interp1d( altitude_edges, pressure_edges )
    pressure_apcemm = fn_z_to_p( altitude_apcemm )
 
    # Get time of current apcemm simulation
    cur_datetime = flight_datetime + dt.timedelta( hours=int(hh), minutes=int(mm) )
    
    # Redefine the sizes based on the collapsed data
    nrow, ncol = np.shape( icenumber_vol_sum )
    areaCell = (Y[2] - Y[1])*(X_sum[2] - X_sum[1])
    
    # For later usage
    dx_sum = X_sum[2] - X_sum[1]
    
    # Storage for RF values
    LW_RF = np.zeros( ncol )
    SW_RF = np.zeros( ncol )
    Net_RF = np.zeros( ncol )
   
    # Define data which will be needed by RRTM
    #julian_day = 1721424.5 + cur_datetime.toordinal()
    # RRTM only wants the day of the year (1-366)
    julian_day = cur_datetime.timetuple().tm_yday

    # Convert to formats needed by RRTM
    emissivity = np.array([emissivity_val])

    # Loop over columns
    #print( sza, emissivity, albnirdf, albnirdr, albvisdf, albvisdr, tropopause )
    for icol in range(ncol):
        
        # Extract information for a specific column
        icenumber_vol_col = icenumber_vol_sum[:,icol]
        icemass_vol_col = icemass_vol_sum[:,icol]
        effradius_col = effradius_sum[:,icol]
        if np.sum(icemass_vol_col)*areaCell <= min_icemass: # 1E-5:
            continue
        if verbose:
            print( '       ', icol, np.sum(icemass_vol_col)*areaCell )
       
        # Map APCEMM data onto a full column grid for RRTM
        pressure_edges_rrtm, IWC_rrtm, reff_rrtm, cldfr_rrtm, cliqwp_rrtm, cicewp_rrtm = \
                convertConditions( icemass_vol_col, icenumber_vol_col, effradius_col, \
                                   altitude_apcemm, altitude_edges, fn_z_to_p, cldfr, \
                                   clwp, ciwp )
        if ( not monotonic( pressure_edges_rrtm ) ):
            for idx, val in enumerate( pressure_edges_rrtm[:-1] ):
                print( val, IWC_rrtm[idx], reff_rrtm[idx] )
            print( apcemm_data_file )
            #sys.exit(0)
            raise ValueError('Non-monotonic pressure edges')

        # File names to edit
        #rrtm_lw_binary_path = 'rrtmg_lw_wcomments'
        #rrtm_sw_binary_path = 'rrtmg_sw_wcomments'
        file_tag = 't{:02d}{:02d}_c{:03d}'.format(int(hh),int(mm),icol)
        file_lw_in   = os.path.join(ref_dir,'input_rrtm_lw_template')
        file_lw_out  = os.path.join(apcemm_folder_rrtm,'lw_input_{:s}'.format(file_tag))
        file_sw_in   = os.path.join(ref_dir,'input_rrtm_sw_template')
        file_sw_out  = os.path.join(apcemm_folder_rrtm,'sw_input_{:s}'.format(file_tag))
        file_cld_in  = os.path.join(ref_dir,'incld_rrtm_lw_template')
        file_cld_out = os.path.join(apcemm_folder_rrtm,'cld_input_{:s}'.format(file_tag))
        
        file_lw_out_clr, file_lw_out_cld = edit_lw_input( file_lw_in, file_lw_out, pressure, temperature,
                                                          pressure_edges_rrtm, relative_humidity, emissivity )
        file_sw_out_clr, file_sw_out_cld = edit_sw_input( file_sw_in, file_sw_out, pressure, temperature, relative_humidity, \
                                                          pressure_edges_rrtm, emissivity, julian_day, sza, albnirdf, albnirdr, albvisdf, albvisdr )
        file_cld_out_clr, file_cld_out_cld = edit_cld_input( file_cld_in, file_cld_out,
                                                             IWC_rrtm, reff_rrtm, cldfr_rrtm,
                                                             cliqwp_rrtm, cicewp_rrtm )
        
    return ncol, dx_sum
    
    #return np.dot(Net_RF, width_sum), np.dot(LW_RF, width_sum), np.dot(SW_RF, width_sum), time #, sza, emissivity, cldfr, cliqwp, cicewp

# Map APCEMM data onto the meteorological data by introducing an additional
# layer which includes the contrail
def convertConditions( icemass_vol_col, icenumber_vol_col, effradius_col, \
                       altitude_apcemm, altitude_edges, fn_z_to_p, cldfr, \
                       cliqwp, cicewp ):
    # For the below, n_apcemm is the number of APCEMM layers and
    # n_met is the number of meteorological layers:
    # icemass_vol_col[n_apcemm]       Ice mass (kg/m3)
    # icenumber_vol_col[n_apcemm]     Number of ice particles (#/m3)
    # effradius_col[n_apcemm]         Effective radius (m?)
    # altitude_apcemm[n_apcemm+1]     Altitude at APCEMM layer edges (m)
    # altitude_edges[n_met+1]         Altitude at met data layer edges (m)
    # fn_z_to_p                       Function to convert altitude to pressure (m)
    # cldfr[n_met]                    Cloud fraction (0-1, unitless)
    # cliqwp[n_met]                   Cloud liquid water path (g/cm2)
    # cicewp[n_met]                   Cloud ice water path (g/cm2)
    
    # Get 1% and 99% points and select lowest and highest
    Ylom, Yhim = aps.getAPCEMM_upperlower( icemass_vol_col, altitude_apcemm )
    Ylon, Yhin = aps.getAPCEMM_upperlower( icenumber_vol_col, altitude_apcemm )
    Ylo, Yhi = ( min( Ylom, Ylon ), max( Yhim, Yhin ) )
    if Ylo > Yhi: # Weird bug?
        print( 'Ylo > Yhi???')
        altitude_edges_rrtm = altitude_edges
        altitude_rrtm = 0.5*( altitude_edges_rrtm[1:] + altitude_edges_rrtm[:-1] )
        pressure_edges_rrtm = fn_z_to_p( altitude_edges_rrtm )
        IWC_rrtm = np.zeros( len(altitude_rrtm) )
        reff_rrtm = np.zeros( len(altitude_rrtm) )
        return pressure_edges_rrtm, IWC_rrtm, reff_rrtm, cldfr, cliqwp, cicewp
    
    # Find layer below Ylo and above Yhi
    met_lo_idx = len(altitude_edges) - np.argmax( altitude_edges[::-1]<Ylo ) - 1
    met_hi_idx = np.argmax( altitude_edges>Yhi )
    
    # Add layer to altitude_edges
    altitude_edges_rrtm = np.insert( altitude_edges, met_hi_idx, Yhi, 0 )
    altitude_edges_rrtm = np.insert( altitude_edges_rrtm, met_lo_idx+1, Ylo, 0 )
    altitude_rrtm = 0.5*( altitude_edges_rrtm[1:] + altitude_edges_rrtm[:-1] )
    
    # Estimate new pressure edges
    pressure_edges_rrtm = fn_z_to_p( altitude_edges_rrtm )
    pressure_rrtm = 0.5*( pressure_edges_rrtm[1:] + pressure_edges_rrtm[:-1] )
    
    # Add layers to cloud data
    cldfr_rrtm = np.insert( cldfr, met_hi_idx, cldfr[met_hi_idx], 0 )
    cldfr_rrtm = np.insert( cldfr_rrtm, met_lo_idx+1, cldfr_rrtm[met_lo_idx+1], 0 )
    cliqwp_rrtm = np.insert( cliqwp, met_hi_idx, cliqwp[met_hi_idx], 0 )
    cliqwp_rrtm = np.insert( cliqwp_rrtm, met_lo_idx+1, cliqwp[met_lo_idx+1], 0 )
    cicewp_rrtm = np.insert( cicewp, met_hi_idx, cliqwp[met_hi_idx], 0 )
    cicewp_rrtm = np.insert( cicewp_rrtm, met_lo_idx+1, cliqwp[met_lo_idx+1], 0 )
    
    # Restate the indices of the edges of the contrail in MERRA-2
    met_lo_idx += 1
    met_hi_idx += 1
    
    # Loop over each inserted layer and calculate IWC and effective radius
    f_zoh_icevol = gen_zoh( altitude_apcemm, icemass_vol_col ) # Use to integrate ice mass
    f_zoh_deff = gen_zoh( altitude_apcemm, 2*effradius_col ) # Use to integrate ice number
    IWC_rrtm = np.zeros( len(altitude_rrtm) )
    reff_rrtm = np.zeros( len(altitude_rrtm) )
    for layer in np.arange( met_lo_idx, met_hi_idx ):
        Z_layer_low = altitude_edges_rrtm[layer]
        Z_layer_high = altitude_edges_rrtm[layer+1]
        IWC_temp = summation( icemass_vol_col, altitude_apcemm, Z_layer_low, Z_layer_high )
        if IWC_temp < 1E-10:
            IWC_rrtm[layer] = 0
            reff_rrtm[layer] = 0
        else:
            IWC_rrtm[layer] = IWC_temp # sint.quad( f_zoh_icevol, Z_layer_low, Z_layer_high )[0]
            reff_rrtm[layer] = summation( icemass_vol_col*2*effradius_col, altitude_apcemm, \
                                          Z_layer_low, Z_layer_high )/IWC_temp
    
    return pressure_edges_rrtm, IWC_rrtm, reff_rrtm, cldfr_rrtm, cliqwp_rrtm, cicewp_rrtm

# Load in RRTM output
def readRRTMOutput( folderpath, file_lw_clr, file_sw_clr, 
                    file_lw_cld, file_sw_cld, p_tropopause=200, maxlevs=200 ):
    
    # File names
    #file_lw_clr = '/OUTPUT-RRTM_lw_clr'
    #file_sw_clr = '/OUTPUT-RRTM_sw_clr'
    #file_lw_cld = '/OUTPUT-RRTM_lw_cld'
    #file_sw_cld = '/OUTPUT-RRTM_sw_cld'
    
    # Load data
    #nlevs = len(pressure_edges_rrtm)
    file_lw_clr = read_LW( os.path.join(folderpath,file_lw_clr), maxlevs)
    file_sw_clr = read_SW( os.path.join(folderpath,file_sw_clr), maxlevs)
    file_lw_cld = read_LW( os.path.join(folderpath,file_lw_cld), maxlevs)
    file_sw_cld = read_SW( os.path.join(folderpath,file_sw_cld), maxlevs)
    
    # Find values at tropopause
    row_lw_clr = file_lw_clr.loc[ file_lw_clr['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    row_sw_clr = file_sw_clr.loc[ file_sw_clr['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    row_lw_cld = file_lw_cld.loc[ file_lw_cld['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    row_sw_cld = file_sw_cld.loc[ file_sw_cld['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    
    # Calculate cloud radiative effect
    LW_RF = row_lw_clr['NET FLUX'] - row_lw_cld['NET FLUX']
    if (not isinstance( row_sw_clr['NET FLUX'], numbers.Number )) | \
       (not isinstance( row_sw_cld['NET FLUX'], numbers.Number )):
        SW_RF = 0
    elif (np.isnan( row_sw_cld['NET FLUX'] )) | (np.isnan( row_sw_cld['NET FLUX'] )):
        SW_RF = 0
    else:
        SW_RF = row_sw_clr['NET FLUX'] - row_sw_cld['NET FLUX']
    Net_RF = LW_RF - SW_RF
    
    return Net_RF, LW_RF, SW_RF

def read_LW( filename, maxlevs=1000 ):
    
    # Open file and read lines
    f = open( filename, "r" )
    lines = f.readlines()
   
    nlevs = maxlevs
 
    # Loop over lines
    ii = 0
    level = np.zeros( nlevs )
    pressure = np.zeros( nlevs )
    temperature = np.zeros( nlevs )
    upward_flux = np.zeros( nlevs )
    downward_flux = np.zeros( nlevs )
    net_flux = np.zeros( nlevs )
    heating_rate = np.zeros( nlevs )
    for line in lines:
        
        # Split string
        split_line = line.split()
        if len(split_line)==0:
            break
        
        # Get data and add to dataframe
        cur_data = np.array( split_line )
        if cur_data[0].isnumeric():
            level[ii] = cur_data[0]
            pressure[ii] = cur_data[1]
            temperature[ii] = cur_data[2]
            upward_flux[ii] = cur_data[3]
            downward_flux[ii] = cur_data[4]
            net_flux[ii] = cur_data[5]
            heating_rate[ii] = cur_data[6]
            ii += 1
    
    df_lw = pd.DataFrame( { 'LEVEL':level[:ii], 'PRESSURE':pressure[:ii], \
                            'TEMPERATURE':temperature[:ii], 'UPWARD FLUX':upward_flux[:ii], \
                            'DOWNWARD FLUX':downward_flux[:ii], 'NET FLUX':net_flux[:ii], \
                            'HEATING RATE':heating_rate[:ii] } )
    
    return df_lw

def read_SW( filename, maxlevs=1000 ):
    
    nlevs = maxlevs

    # Open file and read lines
    f = open( filename, "r" )
    lines = f.readlines()
    lines = lines[2:]
    
    # Create empty df
    headings = [ 'LEVEL', 'PRESSURE', 'UPWARD FLUX', 'DIFDOWN FLUX', 'DIRDOWN FLUX', \
                 'DOWNWARD FLUX', 'NET FLUX', 'HEATING RATE' ]
    df_sw = pd.DataFrame( columns=headings )
    
    # Loop over lines
    ii = 0
    level = np.zeros( nlevs )
    pressure = np.zeros( nlevs )
    upward_flux = np.zeros( nlevs )
    difdown_flux = np.zeros( nlevs )
    dirdown_flux = np.zeros( nlevs )
    downward_flux = np.zeros( nlevs )
    net_flux = np.zeros( nlevs )
    heating_rate = np.zeros( nlevs )
    for line in lines:
        
        # Split string
        split_line = line.split()
        if len(split_line)==0:
            break
        
        # Get data and add to dataframe
        cur_data = np.array( split_line )
        if cur_data[0].isnumeric():
            level[ii] = cur_data[0]
            pressure[ii] = cur_data[1]
            if '*' in cur_data[2]:
                upward_flux[ii] = np.nan
            else:
                upward_flux[ii] = cur_data[2]
            if '*' in cur_data[3]:
                difdown_flux[ii] = np.nan
            else:
                difdown_flux[ii] = cur_data[3]
            if '*' in cur_data[4]:
                dirdown_flux[ii] = np.nan
            else:
                dirdown_flux[ii] = cur_data[4]
            if '*' in cur_data[5]:
                downward_flux[ii] = np.nan
            else:
                downward_flux[ii] = cur_data[5]
            if '*' in cur_data[6]:
                net_flux[ii] = np.nan
            else:
                net_flux[ii] = cur_data[6]
            if '*' in cur_data[7]:
                heating_rate[ii] = np.nan
            else:
                heating_rate[ii] = cur_data[7]
            ii += 1
    
    df_sw = pd.DataFrame( { 'LEVEL':level[:ii], 'PRESSURE':pressure[:ii], \
                            'UPWARD FLUX':upward_flux[:ii], 'DIFDOWN FLUX':difdown_flux[:ii], \
                            'DIRDOWN FLUX':dirdown_flux[:ii], 'DOWNWARD FLUX':downward_flux[:ii], \
                            'NET FLUX':net_flux[:ii], 'HEATING RATE':heating_rate[:ii] } )
    
    return df_sw

# Edit the LW input file
def edit_lw_input( file_lw_in, file_lw_out, pressure, temperature, pressure_edges_rrtm, relative_humidity, emissivity ):
   
    pressure_hPa = pressure * 0.01
    pressure_edges_rrtm_hPa = pressure_edges_rrtm * 0.01
 
    # Open both files for read/write respectively
    f_in = open(file_lw_in, 'r')
    f_out_clr = open(file_lw_out+'_clr', 'w')
    f_out_cld = open(file_lw_out+'_cld', 'w')
    
    # Read lines
    lines = f_in.readlines()
    # for line in lines:
    #     print(line)
    f_in.close()
    
    # Edit record 1.2
    iline = 4
    emis_line = ''.join( [ '%10.3f' %emissivity[k] for k in range( len( emissivity ) ) ])
    lines[iline] = '%10.3f %1d  %1d'%( temperature[0], 2, 0 ) + emis_line + '\n' 
    #print( 'emissivity: ', emis_line )
    
    # Record 3.1
    iline = 5
    string = '    0       -%02d         1    7    1    0         0                       403.000\n' %(len(pressure_edges_rrtm))
    lines[iline] = string
    
    # Edit 4th line (record 3.2)
    iline = 6
    lines[iline] = '%10.3f%10.3f\n' %( np.max( pressure_edges_rrtm_hPa ), \
                                       np.min( pressure_edges_rrtm_hPa ) )
    
    # Record 3.3B
    iline = 7
    iline_count = 0
    string = ''
    for ilayer, pvalue in enumerate( pressure_edges_rrtm_hPa ):
        string = string + '%10.3f' %( pvalue )
        iline_count+=1
        if iline_count==8:
            string = string + '\n'
            lines.insert(iline, string)
            iline_count = 0
            string = ''
            iline += 1
    if iline_count != 0:
        string = string + '\n'
    lines.insert(iline, string)

    # Record 3.4
    iline += 1
    string = '%5i RH_data\n' % ( -1*( len( pressure ) + 1 ) )
    lines[iline] = string
    
    # Record 3.5 and 3.6
    iline += 1
    string = '%10.3f%10.3f%10.3f     AA   H666666\n' %( 0.0, pressure_edges_rrtm_hPa[0], temperature[0] )
    lines[iline] = string
    iline += 1
    lines[iline] = '%10.3e\n' %( relative_humidity[0] )
    iline += 1
    for ilayer, pvalue in enumerate( pressure_hPa ):
        string = '          %10.3f%10.3f     AA   H666666\n' %( pvalue, temperature[ilayer] )
        lines[iline] = string
        iline += 1
        lines[iline] = '%10.3e\n' %( relative_humidity[ilayer] )
        iline += 1

    while iline < len( lines ):
        lines[iline] = ''
        iline += 1
    
#     # Edit record 1.4 (clear sky)
#     iline = 3
#     temp_line = list( lines[iline] )
#     temp_line[94] = '0'
#     lines[iline] = "".join( temp_line )
    
    # Write to clear sky file
    lines_clr = "".join(lines)
    f_out_clr.write(lines_clr)
    f_out_clr.close()
    
#     # Edit record 1.4 (cloudy sky)
#     iline = 3
#     temp_line = list( lines[iline] )
#     temp_line[94] = '2'
#     lines[iline] = "".join( temp_line )
    
    # Write to cloudy sky file
    lines_cld = "".join(lines)
    f_out_cld.write(lines_cld)
    f_out_cld.close()
    
    return file_lw_out+'_clr', file_lw_out+'_cld'

# Edit the SW input file
def edit_sw_input( file_sw_in, file_sw_out, pressure, temperature, relative_humidity, pressure_edges_rrtm, emissivity, julian_day, sza, albnirdf, albnirdr, albvisdf, albvisdr ):
    # pressure[:]              mid points (Pa?) on which RH and temperature are defined
    # temperature[:]           temperature (K) in each cell
    # relative_humidity[:]     relative humidity (%?) in each cell
    # pressure_edges_rrtm[:]   pressure (Pa?) defining how RRTM will be run (seems odd that this is different from the other vector?)
    # julian_day               integer julian day on which the calculation is performed
    # sza                      solar zenith angle (degrees)
    # albnirdf                 surface albedo (fraction) for diffuse, near-IR radiation
    # albnirdr                 surface albedo (fraction) for direct, near-IR radiation
    # albvisdf                 surface albedo (fraction) for diffuse, visible radiation
    # albvisdr                 surface albedo (fraction) for direct, visible radiation
 
    pressure_hPa = pressure * 0.01
    pressure_edges_rrtm_hPa = pressure_edges_rrtm * 0.01

    # Open both files for read/write respectively
    f_in = open(file_sw_in, 'r')
    f_out_clr = open(file_sw_out+'_clr', 'w')
    f_out_cld = open(file_sw_out+'_cld', 'w')
    
    # Read lines
    lines = f_in.readlines()
    # for line in lines:
    #     print(line)
    f_in.close()
    
    # Record 1.2.1
    iline = 5
    string = '            %3d   %6.4f\n' %( julian_day, sza )
    lines[iline] = string
    
    # Record 1.4
    iline = 6
    string = '           1  0 ' + ('%5.3f' %( emissivity[-1] ) ).lstrip('0') + '\n'

    lines[iline] = string
    
    # New record 1.5 (Albedo values)
    iline = 7
    string = '           1  %5.3f%5.3f%5.3f%5.3f\n' %(albnirdf, albnirdr, \
                                                      albvisdf, albvisdr)
    lines[iline] = string
    
    # Record 3.1
    iline = 8
    string = '    0       -%02d         1    7    1    0         0                       403.000\n' %(len(pressure_edges_rrtm))
    lines[iline] = string
    
    # Edit 9th line (record 3.2)
    iline = 9
    lines[iline] = '%10.3f%10.3f\n' %( np.max( pressure_edges_rrtm_hPa ), \
                                       np.min( pressure_edges_rrtm_hPa ) )
    
    # Record 3.3B
    iline = 10
    iline_count = 0
    string = ''
    for ilayer, pvalue in enumerate( pressure_edges_rrtm_hPa ):
        string = string + '%10.3f' %( pvalue )
        iline_count+=1
        if iline_count==8:
            string = string + '\n'
            lines.insert(iline, string)
            iline_count = 0
            string = ''
            iline += 1
    if iline_count != 0:
        string = string + '\n'
    lines.insert(iline, string)
    
    # Record 3.4
    iline += 1
    string = '%5i RH_data\n' % ( -1*( len( pressure ) + 1 ) )
    lines[iline] = string

    # Record 3.5 and 3.6
    iline += 1
    string = '%10.3f%10.3f%10.3f     AA   H666666\n' %( 0.0, pressure_edges_rrtm_hPa[0], temperature[0] )
    lines[iline] = string
    iline += 1
    lines[iline] = '%10.3e\n' %( relative_humidity[0] )
    iline += 1
    for ilayer, pvalue in enumerate( pressure_hPa ):
        string = '          %10.3f%10.3f     AA   H666666\n' %( pvalue, temperature[ilayer] )
        lines[iline] = string
        iline += 1
        lines[iline] = '%10.3e\n' %( relative_humidity[ilayer] )
        iline += 1
    
    while iline < len( lines ):
        lines[iline] = ''
        iline += 1
    
    # Write to cloudy sky file
    lines_clr = "".join(lines)
    f_out_clr.write(lines_clr)
    f_out_clr.close()
    
    # Write to cloudy sky file
    lines_cld = "".join(lines)
    f_out_cld.write(lines_cld)
    f_out_cld.close()
    
    return file_sw_out+'_clr', file_sw_out+'_cld'

# Edit the cloud input file
def edit_cld_input( file_cld_in, file_cld_out, IWC_rrtm, reff_rrtm, cldfr, cliqwp, cicewp ):
    # IWC_rrtm[:]              vector of ice water content in each grid cell on the RRTM levels
    
    # Natural cloud data
    REL_DEF = 14.2
    REI_DEF=24.8
    
    # Open both files for read/write respectively
    f_in = open(file_cld_in, 'r')
    f_out_clr = open(file_cld_out+'_clr', 'w')
    f_out_cld = open(file_cld_out+'_cld', 'w')
    
    # Read lines
    lines_clr = f_in.readlines()
    lines_cld = list( lines_clr ) # f_in.readlines()
    f_in.close()
    
    # Record C1.2 (clear)
    iline = 1
    # for layer, cur_cldfr in enumerate(cldfr):
    for layer, cur_coniwp in enumerate(IWC_rrtm):
        # Get cloud data
        cur_cldfr = cldfr[layer]
        cur_cliqwp = cliqwp[layer]
        cur_cicewp = cicewp[layer]
        tot_wp = cur_cliqwp + cur_cicewp
        # Add lines
        if (cur_cldfr>1.0E-03) & (tot_wp>1.0E-03):
            fracice = cur_cicewp/tot_wp
            string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.4f\n' %( \
                                                        layer+1, cur_cldfr, tot_wp, \
                                                        fracice, REI_DEF, REL_DEF, \
                                                        0.0, 0.0 )
            lines_clr.insert( iline, string )
            iline+=1
    
    # Record C1.2 (cloudy)
    iline = 1
    # for layer, cur_cldfr in enumerate(cldfr):
    for layer, cur_coniwp in enumerate(IWC_rrtm):
        # Get cloud data
        cur_cldfr = cldfr[layer]
        cur_cliqwp = cliqwp[layer]
        cur_cicewp = cicewp[layer]
        tot_wp = cur_cliqwp + cur_cicewp
        # Get contrail data
        cur_reff = reff_rrtm[layer] * 1.0E+06
        # Add lines
        if ( (cur_cldfr>1.0E-03) & (tot_wp>1.0E-03) ):
            if cur_reff <= 5:
                cur_reff = 6
            elif cur_reff >= 130:
                cur_reff = 129
            fracice = cur_cicewp/tot_wp
            if ( cur_coniwp<=100 ):
                if ( cur_reff<=100 ):
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.4f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
                else:
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.3f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
            else:
                if ( cur_reff<=100 ):
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.3f   %7.4f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
                else:
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.3f   %7.3f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
            lines_cld.insert( iline, string )
            iline+=1
        elif (cur_coniwp>0.0):
            if cur_reff <= 5:
                cur_reff = 6
            elif cur_reff >= 130:
                cur_reff = 129
            fracice = 0.0
            if ( cur_coniwp<=100 ):
                if ( cur_reff<=100 ):
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.4f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
                else:
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.3f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
            else:
                if ( cur_reff<=100 ):
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.3f   %7.4f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
                else:
                    string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.3f   %7.3f\n' %( \
                                                                layer+1, cur_cldfr, tot_wp, \
                                                                fracice, REI_DEF, REL_DEF, \
                                                                cur_coniwp, cur_reff )
            # string = '  %3d    %6.4f %9.3e    %6.4f   %7.4f   %7.4f   %7.4f   %7.4f\n' %( \
            #                                             layer+1, 0.0, 0.0, \
            #                                             1.0, REI_DEF, REL_DEF, \
            #                                             cur_coniwp, cur_reff )
            lines_cld.insert( iline, string )
            iline+=1
        
    # for line in lines:
    #     print(line)
    
    # Write to file (clear)
    lines = "".join(lines_clr)
    f_out_clr.write(lines)
    f_out_clr.close()
    
    # Write to file (cloud)
    lines = "".join(lines_cld)
    f_out_cld.write(lines)
    f_out_cld.close()
    
    return file_cld_out+'_clr', file_cld_out+'_cld'

# Integrate using simple sum between low and high values
def summation( y, x, x_lo, x_hi ):
    # Sum y within x_lo to x_hi
    
    # Get lower and upper indices
    C0 = x[0]; delE = x[1] - x[0]
    i_lo = int( np.floor( (x_lo-C0)/delE - 0.5 ) )
    i_hi = int( np.floor( (x_hi-C0)/delE - 0.5 ) )
    
    # Integrated
    integrated_value = np.sum( y[i_lo:i_hi] )*delE
    
    return integrated_value

# Generate a zero order hold function
def gen_zoh( xdata, ydata ):
    
    # Ensure x_data is decreasing
    if xdata[1]>=xdata[0]:
        xdata = np.flipud( xdata )
        ydata = np.flipud( ydata )
    
    # Get edges
    xdata_edge = np.zeros( len(xdata) )
    xdata_edge[:-1] = 0.5*( xdata[1:] + xdata[:-1] )
    xdata_edge[-1] = xdata[-1] - (xdata_edge[-2]-xdata[-1])
    
    func_zoh = sipl.interp1d( x=xdata_edge, y=ydata, kind='zero', fill_value='extrapolate' )
    
    return func_zoh

#def main_function(ts_aerosol_folder, z_flight, apcemm_met_input_file):
#    # Load input data
#    nc_input = nc.Dataset( apcemm_met_input_file )
#    # evaporation_depth = nc_input['evaporation_depth'][:]
##     if ( evaporation_depth==0 ) | ( np.isnan( evaporation_depth ) ):
##         print( '    Evaporation depth = 0... Skipping' )
##         return 0, 0, 0, 0, 0, 0, 0, 0
#    pressure = nc_input['pressure'][:]
#    altitude = nc_input['altitude'][:]*1.0E+03
#    temperature = nc_input['temperature'][:]
#    relative_humidity = nc_input['relative_humidity'][:]
#    
#    # Calculate layer edges for altitudes
#    altitude_edges = np.zeros( len(altitude) + 1 )
#    altitude_edges[1:-1] = 0.5*( altitude[1:] + altitude[:-1] )
#    altitude_edges[-1] = 2*altitude[-1] - altitude_edges[-2]
#    
#    # Calculate layer edges for pressures
#    pressure_edges = np.zeros( len(pressure) + 1 )
#    pressure_edges[1:-1] = 10**(0.5*( np.log10(pressure[1:]) + np.log10(pressure[:-1]) ))
#    pressure_edges[0] = 10**(2*np.log10(pressure[0]) - np.log10(pressure_edges[1]))
#    pressure_edges[-1] = 10**(2*np.log10(pressure[-1]) - np.log10(pressure_edges[-2]))
#
#    # Flight time / location
#    year = 2018
#    month = 1
#    day = 20
#    hour = 8
#    flight_latitude = 20.0
#    flight_longitude = -15.0
#
#    flight_datetime = dt.datetime(year, month, day, hour)
#    
#    # Copy rrtm executables to desired location
#    # copyfile( '/net/d08/data/aa681/RRTMG/rrtmg_lw_v4.85_TAMU_v3/column_model/sonde_runs/test_versions/rrtmg_lw_wcomments', \
#    #           'rrtmg_lw_wcomments' )
#    # copyfile( '/net/d08/data/aa681/RRTMG/rrtmg_sw_v4.02_TAMU_v0/column_model/sonde_runs/test_versions/rrtmg_sw_alb', \
#    #           'rrtmg_sw_wcomments' )
#
#    copyfile( '/home/xu990/rrtm_executables/rrtmg_lw_wcomments', \
#              'rrtmg_lw_wcomments' )
#    copyfile( '/home/xu990/rrtm_executables/rrtmg_sw_wcomments', \
#              'rrtmg_sw_wcomments' )
#    # Convert lw to executable
#    bashCommand = 'chmod +x rrtmg_sw_wcomments'
#    process = subprocess.Popen( bashCommand.split(), stdout=subprocess.PIPE )
#    output, error = process.communicate()
#    
#    # Get aerosol files
#    aerosol_files = np.sort( glob(f'{ts_aerosol_folder}/ts_aerosol*' ) )
#    aerosol_files = aerosol_files
#    nfiles = len( aerosol_files )
#    if nfiles==0:
#        print( '    No aerosol files... Skipping' )
#        return 0, 0, 0, 0, 0, flight_datetime, flight_latitude, flight_longitude, 0, 0, 0, 0, 0, 0, 0
#    
#    # Get grid dimensions and zero out output variables
#    nc_apcemm = nc.Dataset( aerosol_files[0] )
#    X, Y, areaCell = aps.getAPCEMM_grid( nc_apcemm )
#    dY = Y[2] - Y[1]
#    number2sum = 16
#    approach = 0
#    ncol_model = len(X)
#    ncol = ncols_apcemm_colsums( ncol_model, approach=approach, number2sum=number2sum )
#    nrow = len(Y)
#    if met_data_flag == 'MERRA2':
#        n_alt = 72
#        n_emis= 1
#    elif met_data_flag == 'ERA5':
#        n_alt = 28
#        n_emis= 1
#    else:
#        raise ValueError('Invalid met_data_flag: %s' %met_data_flag)
#    Net_RF = np.zeros(nfiles) 
#    LW_RF = np.zeros(nfiles)
#    SW_RF = np.zeros(nfiles)
#    time = np.zeros(nfiles)
#    sza = np.zeros( nfiles )
#    emissivity = np.zeros( (n_emis, nfiles) )
#    icemass = np.zeros( nfiles )
#    icenumber = np.zeros( nfiles )
#    cldfr = np.zeros( (n_alt, nfiles) )
#    cliqwp = np.zeros( (n_alt, nfiles) )
#    cicewp = np.zeros( (n_alt, nfiles) )
#    
#    # Loop over aerosol files
#    for ifile, file in enumerate( aerosol_files ):
#        
#        Net_RF[ifile], LW_RF[ifile], SW_RF[ifile], time = APCEMM2RRTM_V2( file, pressure, altitude, temperature, \
#                                                                            relative_humidity, z_flight, flight_latitude, \
#                                                                            flight_longitude, flight_datetime, number2sum, approach)
#        
#    out_filename = ts_aerosol_folder + "/rrtm_output.csv"
#    out_df = pd.DataFrame(data={"Net_RF": Net_RF, "LW_RF": LW_RF, "SW_RF": SW_RF, "time": time})
#    out_df.to_csv(out_filename)


def non_increasing(L):
    return all(x>y for x, y in zip(L, L[1:]))

def non_decreasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def monotonic(L):
    return non_increasing(L) or non_decreasing(L)

#apcemm_met_input_file ='/home/xu990/contrail_advection_pipeline/traces/traces/20180120_07_37_185_FDX52_met_trace_y2018_contrail_c0_000000.nc'
#ts_aerosol_folder = "/home/xu990/contrail_lidar_comparisons/APCEMM_const_rhi/20180709_19_14_36_N346L/EI_soot_0.01"
#z_flight = 10800.0
#main_function(ts_aerosol_folder, z_flight, apcemm_met_input_file)

def calc_sza(lat,lon,curr_dt):
    return 90.0 - pys.get_altitude_fast(lat,lon,curr_dt.replace(tzinfo=dt.timezone.utc))

def APCEMM_RF(ts_dir,z_flight,flight_datetime,lat_vec,lon_vec,
              sw_bin,lw_bin,ref_dir,
              altitude_edges,fn_z_to_p,temperature_array,rh_array,
              dt=None,dt_max=None,
              number2sum=16,approach=0,
              min_icemass=1.0e-5,verbose=False):
             
    from run_RRTM import run_directory
    from time import time
    from datetime import timedelta

    # Step 1: Generate RRTM input files
    if dt is None:
        dt = timedelta(minutes=10)
    if dt_max is None:
        dt_max = timedelta(hours=24)
    
    dt_base = timedelta(hours=0)
    dt_curr = dt_base

    # Need absolute path to the reference directory, to be safe
    ref_dir_abs = os.path.abspath(ref_dir)

    dx_data = {}
    ncol_data = {}
    i_time = 0
    # Dummy values - come back to this
    emissivity = 0.8
    albnirdf = 0.3
    albnirdr = 0.3
    albvisdf = 0.3
    albvisdr = 0.3

    sza_vec = []
    while dt_curr < dt_max:
        total_sec = dt_curr.total_seconds()
        hh = int(np.floor(total_sec/3600.0))
        mm = int(np.mod(total_sec/60.0,60))
        tstamp = '{:02d}{:02d}'.format(hh,mm)
        f = 'ts_aerosol_case0_{:s}.nc'.format(tstamp)
        f_APCEMM = os.path.join(ts_dir,f)
        if not os.path.isfile(f_APCEMM):
            break
        # Allow for time-varying or constant T and RH columns
        if temperature_array.ndim == 2:
            temperature = temperature_array[i_time,:]
            rh          = rh_array[i_time,:]
        else:
            temperature = temperature_array[:]
            rh          = rh_array[:]
        if np.isscalar(lat_vec):
            lat = lat_vec
            lon = lon_vec
        else:
            lat = lat_vec[i_time]
            lon = lon_vec[i_time]
        sza = calc_sza(lat,lon,dt_curr + flight_datetime)
        sza_vec.append(sza)
        ncol, dx = APCEMM2RRTM_V2(f_APCEMM,z_flight,
                                  flight_datetime,number2sum,
                                  approach,altitude_edges,fn_z_to_p,
                                  temperature,rh,ref_dir=ref_dir_abs,
                                  emissivity=emissivity,albnirdf=albnirdf,
                                  albnirdr=albnirdr,albvisdf=albvisdf,
                                  albvisdr=albvisdr,sza=sza,
                                  verbose=False)
        dx_data[tstamp] = dx
        ncol_data[tstamp] = ncol
        dt_curr += dt

    # Step 2: Run RRTM
    t_start = time()
    rrtm_dir = os.path.join(ts_dir,'rrtm')
    run_directory(rrtm_dir,sw_bin=sw_bin,lw_bin=lw_bin,verbose=verbose)
    t_stop = time()
    if verbose:
        print('Completed calculations in {:.1f} seconds'.format(t_stop-t_start))

    # Step 3: Calculate forcing 
    f_list = [x for x in os.listdir(rrtm_dir) if x.startswith('sw_output_t') and x.endswith('_clr')]
    rf = {'net': [], 'sw': [], 'lw': []}
    rf_2D = {'net': [], 'sw': [], 'lw': [], 'width': []}
    dt_curr = dt_base
    t = []
    while dt_curr < dt_max:
        total_sec = dt_curr.total_seconds()
        hh = int(np.floor(total_sec/3600.0))
        mm = int(np.mod(total_sec/60.0,60))
        tstamp = '{:02d}{:02d}'.format(hh,mm)
        f_list_mini = [x for x in f_list if 't' + tstamp in x]
        ncol = len(f_list_mini)
        if ncol == 0:
            break
        net = 0
        lw = 0
        sw = 0
        col_width = dx_data[tstamp]
        rf_2D['width'].append(col_width)
        ncol = len(f_list_mini)
        rf_2D['net'].append(np.zeros(ncol))
        rf_2D['lw'].append(np.zeros(ncol))
        rf_2D['sw'].append(np.zeros(ncol))
        for icol, f in enumerate(f_list_mini):
            column = int(f.split('_')[3][1:])
            f_lw_clr = 'lw_output_t{:s}_c{:03d}_clr'.format(tstamp,column)
            f_sw_clr = 'sw_output_t{:s}_c{:03d}_clr'.format(tstamp,column)
            f_lw_cld = 'lw_output_t{:s}_c{:03d}_cld'.format(tstamp,column)
            f_sw_cld = 'sw_output_t{:s}_c{:03d}_cld'.format(tstamp,column)
            net_col, lw_col, sw_col = readRRTMOutput(rrtm_dir,f_lw_clr,f_sw_clr,f_lw_cld,f_sw_cld)
            # Store the "full" data
            rf_2D['net'][-1][icol] = net_col
            rf_2D['lw'][-1][icol] = lw_col
            rf_2D['sw'][-1][icol] = sw_col
            # Aggregate from W/m2 to W/m
            net += net_col * col_width
            lw += lw_col * col_width
            sw += sw_col * col_width
        rf['net'].append(net)
        rf['lw'].append(lw)
        rf['sw'].append(sw)
        t.append(flight_datetime + dt_curr)
        dt_curr += dt

    aux_data = {'rf_2D': rf_2D, 'sza': sza_vec}
    return t, rf, aux_data
