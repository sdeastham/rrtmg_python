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

try:
    from pyLRT import RadTran, get_lrt_folder
    import copy
    libRadtran_present = True
except:
    libRadtran_present = False

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
                    cldfr=None,clwp=None,ciwp=None,
                    use_mca_lw=True,use_mca_sw=True,skip_sw=False,
                    use_libRadtran=False):
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
    # use_mca_lw                      Use MCA for cloud overlap on LW calculations? (See comment*)
    # use_mca_sw                      Use MCA for cloud overlap on SW calculations? (See comment*)
    # skip_sw                         Only create LW input files
    # use_libRadtran                  Use libRadtran instead of RRTMG
    # * A comment in the LW code mentions that McICA is not recommended for use with the TAMU updates,
    #   but an email conversation between Akshat Agarwal and TAMU in 2021 indicated that there was no
    #   physical reason to not implement the changes in McICA. It is not yet clear if the latest
    #   version of the code includes the TAMU edits in the LW (or SW) McICA codes.

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

    # Time is extracted from the file _name_
    hh, mm = aps.getAPCEMM_time( apcemm_data_file )
    time = int(hh) + int(mm)/60
    if verbose:
         print( '    %s:%s' %( hh, mm ) )
    
    # Get apcemm data from netCDF contents
    with nc.Dataset( apcemm_data_file, 'r' ) as nc_apcemm:
        X, Y, areaCell = aps.getAPCEMM_grid( nc_apcemm )
        dY = Y[2] - Y[1]

        # For later use
        y_vec = Y.copy()
        yb_vec = np.linspace(y_vec[0] - (dY/2.0),y_vec[-1] + (dY/2.0),Y.size)
    
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
    areaCell = (Y[2] - Y[1])*width_sum[0]
    
    # For later usage
    #dx_sum = X_sum[2] - X_sum[1]
    dx_sum = width_sum[0]
    
    # For later use
    x_vec = X_sum.copy()
    xb_vec = np.linspace(x_vec[0] - (dx_sum/2.0),x_vec[-1] + (dx_sum/2.0),Y.size)
    
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

    #import matplotlib.pyplot as plt
    #f, ax = plt.subplots()
    #ax.pcolormesh(icenumber_vol_sum)
    
    # Loop over columns
    #print( sza, emissivity, albnirdf, albnirdr, albvisdf, albvisdr, tropopause )
    aux_data = []
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
        pressure_edges_rrtm, iwp_rrtm, reff_rrtm, cldfr_rrtm, cliqwp_rrtm, cicewp_rrtm, altitude_edges_rrtm = \
                convertConditions( icemass_vol_col, icenumber_vol_col, effradius_col, \
                                   altitude_apcemm, altitude_edges, fn_z_to_p, cldfr, \
                                   clwp, ciwp )
        if ( not monotonic( pressure_edges_rrtm ) ):
            for idx, val in enumerate( pressure_edges_rrtm[:-1] ):
                print( val, iwp_rrtm[idx], reff_rrtm[idx] )
            print( apcemm_data_file )
            #sys.exit(0)
            raise ValueError('Non-monotonic pressure edges')

        if use_libRadtran:
            # Generate the liquid and ice cloud vectors
            # These are top-down, of course
            # Convert ice water path (kg/m2) to ice water content (g/m3)
            iwc_rrtm = 1.0e3 * iwp_rrtm / np.diff(altitude_edges_rrtm)
            # Convert m to km
            z_vec = altitude_edges_rrtm[::-1]/1000.0
            iwc = np.zeros(z_vec.shape)
            iwc[1:] = iwc_rrtm[::-1]
            r_eff = np.zeros(z_vec.shape)
            r_eff[1:] = reff_rrtm[::-1] * 1.0e6 # Convert m to um
            # Remove unnecessary layers
            nz = np.nonzero(iwc)[0]
            # Start from the zero above the first non-zero value
            first_idx = nz[0] - 1
            # Add one because we do actually want that last value
            last_idx  = nz[-1] + 1
            contrail_data = {'z':   np.array(z_vec[first_idx:last_idx]),
                             'iwc': np.array(iwc[first_idx:last_idx]), # g/m3
                             're':  np.array(r_eff[first_idx:last_idx])}

            # Prep the atmosphere description from the meteorological data
            # Expect Z in km, p in hPa, and T in K
            # Input should be oriented with positive -> down
            altitude_mid = 0.5 * (altitude_edges[1:] + altitude_edges[:-1])
            atmosphere = {'z': np.flip(altitude_edges * 1.0e-3),
                          'p': np.flip(fn_z_to_p(altitude_edges) * 0.01),
                          'T': np.flip(np.interp(altitude_edges,altitude_mid,temperature)),
                          'extrap': False,
                          'ref': None}
            #atmosphere = None
            slrt_clear, slrt_cloudy, tlrt_clear, tlrt_cloudy = setup_LRT(
                    ice_data=contrail_data,emissivity=emissivity,albedo=albvisdr,
                    sza=sza,atmosphere=atmosphere,lrt_data_path=None,env=None)
            aux_data.append({'slrt_clear':  slrt_clear,
                             'slrt_cloudy': slrt_cloudy,
                             'tlrt_clear':  tlrt_clear,
                             'tlrt_cloudy': tlrt_cloudy,
                             'sza': sza})
        else:
            # File names to edit
            file_tag = 't{:02d}{:02d}_c{:03d}'.format(int(hh),int(mm),icol)
            file_lw_in   = os.path.join(ref_dir,'input_rrtm_lw_template')
            file_lw_out  = os.path.join(apcemm_folder_rrtm,'lw_input_{:s}'.format(file_tag))
            file_sw_in   = os.path.join(ref_dir,'input_rrtm_sw_template')
            file_sw_out  = os.path.join(apcemm_folder_rrtm,'sw_input_{:s}'.format(file_tag))
            file_cld_in  = os.path.join(ref_dir,'incld_rrtm_lw_template')
            file_cld_out = os.path.join(apcemm_folder_rrtm,'cld_input_{:s}'.format(file_tag))

            file_cld_out_clr, file_cld_out_cld,any_cloud = edit_cld_input( file_cld_in, file_cld_out,
                                                                           iwp_rrtm, reff_rrtm, cldfr_rrtm,
                                                                           cliqwp_rrtm, cicewp_rrtm )
            file_lw_out_clr, file_lw_out_cld = edit_lw_input( file_lw_in, file_lw_out, pressure, temperature,
                                                              pressure_edges_rrtm, relative_humidity, emissivity, use_mca_lw,any_cloud )
            if not skip_sw:
                file_sw_out_clr, file_sw_out_cld = edit_sw_input( file_sw_in, file_sw_out, pressure, temperature, relative_humidity, 
                                                                  pressure_edges_rrtm, emissivity, julian_day, sza, albnirdf, albnirdr,
                                                                  albvisdf, albvisdr,use_mca_sw,any_cloud)
            
            aux_data = None
    
    return ncol, dx_sum, x_vec, xb_vec, y_vec, yb_vec, aux_data
    
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
        iwp_rrtm = np.zeros( len(altitude_rrtm) )
        reff_rrtm = np.zeros( len(altitude_rrtm) )
        return pressure_edges_rrtm, iwp_rrtm, reff_rrtm, cldfr, cliqwp, cicewp
    
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
    iwp_rrtm = np.zeros( len(altitude_rrtm) )
    reff_rrtm = np.zeros( len(altitude_rrtm) )
    for layer in np.arange( met_lo_idx, met_hi_idx ):
        Z_layer_low = altitude_edges_rrtm[layer]
        Z_layer_high = altitude_edges_rrtm[layer+1]
        # Ice water path in kg/m2
        iwp_temp = summation( icemass_vol_col, altitude_apcemm, Z_layer_low, Z_layer_high )
        if iwp_temp < 1E-10:
            iwp_rrtm[layer] = 0
            reff_rrtm[layer] = 0
        else:
            iwp_rrtm[layer] = iwp_temp # sint.quad( f_zoh_icevol, Z_layer_low, Z_layer_high )[0]
            reff_rrtm[layer] = summation( icemass_vol_col*2*effradius_col, altitude_apcemm, \
                                          Z_layer_low, Z_layer_high )/iwp_temp
    
    return pressure_edges_rrtm, iwp_rrtm, reff_rrtm, cldfr_rrtm, cliqwp_rrtm, cicewp_rrtm, altitude_edges_rrtm

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
    row_lw_cld = file_lw_cld.loc[ file_lw_cld['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    
    skip_sw = file_sw_clr is None
    if not skip_sw:
        row_sw_clr = file_sw_clr.loc[ file_sw_clr['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
        row_sw_cld = file_sw_cld.loc[ file_sw_cld['PRESSURE'].sub(p_tropopause).abs().idxmin() ]
    
    # Calculate cloud radiative effect
    LW_RF = row_lw_clr['NET FLUX'] - row_lw_cld['NET FLUX']
    if skip_sw:
        SW_RF = 0.0
    elif (not isinstance( row_sw_clr['NET FLUX'], numbers.Number )) | \
       (not isinstance( row_sw_cld['NET FLUX'], numbers.Number )):
        SW_RF = 0.0
    elif (np.isnan( row_sw_cld['NET FLUX'] )) | (np.isnan( row_sw_cld['NET FLUX'] )):
        SW_RF = 0.0
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
    
    # Shortwave is not always run..
    if not os.path.isfile(filename):
        return None
    
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
def edit_lw_input( file_lw_in, file_lw_out, pressure, temperature, pressure_edges_rrtm, relative_humidity, emissivity, use_mca, any_cloud ):
    # use_mca                  use MCA for clouds (True/False)? WARNING: May not be recommended for TAMU LW code
    # any_cloud                are there any cloud layers present in the contrail-free data?

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
    
    # Record 1.2
    iline_ctrl = 3
    if use_mca:
        mca_int = 1
        cld_int_cloudy = 2
    else:
        mca_int = 0
        cld_int_cloudy = 1
    
    if any_cloud:
        cld_int = cld_int_cloudy
    else:
        cld_int = 0
    # Might need different control lines for the cloudy and clear cases
    ctrl_line_clr = ' HI=0 F4=0 CN=0 AE 0 EM=0 SC=0 FI=0 PL=0 TS=0 AM=1 MG=0 LA=0 OD=0 XS=0   00   00  0 0    0   {:01d}{:1d}\n'.format(mca_int,cld_int)
    ctrl_line_cld = ' HI=0 F4=0 CN=0 AE 0 EM=0 SC=0 FI=0 PL=0 TS=0 AM=1 MG=0 LA=0 OD=0 XS=0   00   00  0 0    0   {:01d}{:1d}\n'.format(mca_int,cld_int_cloudy)
    
    # Record 1.3
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
    lines[iline_ctrl] = ctrl_line_clr
    lines_clr = "".join(lines)
    f_out_clr.write(lines_clr)
    f_out_clr.close()
    
#     # Edit record 1.4 (cloudy sky)
#     iline = 3
#     temp_line = list( lines[iline] )
#     temp_line[94] = '2'
#     lines[iline] = "".join( temp_line )
    
    # Write to cloudy sky file
    lines[iline_ctrl] = ctrl_line_cld
    lines_cld = "".join(lines)
    f_out_cld.write(lines_cld)
    f_out_cld.close()
    
    return file_lw_out+'_clr', file_lw_out+'_cld'

# Edit the SW input file
def edit_sw_input( file_sw_in, file_sw_out, pressure, temperature, relative_humidity, pressure_edges_rrtm, emissivity, julian_day, sza, albnirdf, albnirdr, albvisdf, albvisdr, use_mca, any_cloud ):
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
    # use_mca                  use MCA for clouds (True/False)?
    # any_cloud                are there any cloud layers present in the contrail-free data?
 
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
    
    # Control line - record whether or not to use MCA for cloud overlap
    iline_ctrl = 4
    if use_mca:
        mca_int = 1
        cld_int_cloudy = 2
    else:
        mca_int = 0
        cld_int_cloudy = 1
    
    if any_cloud:
        cld_int = cld_int_cloudy
    else:
        cld_int = 0
    ctrl_line_clr = ' HI=1 F4=1 CN=1 AE 0 EM=0 SC=0 FI=0 PL=0 TS=0 AM=1 MG=1 LA=0    1        00   00  1 0    0   {:01d}{:01d}   00\n'.format(mca_int,cld_int)
    ctrl_line_cld = ' HI=1 F4=1 CN=1 AE 0 EM=0 SC=0 FI=0 PL=0 TS=0 AM=1 MG=1 LA=0    1        00   00  1 0    0   {:01d}{:01d}   00\n'.format(mca_int,cld_int_cloudy)
    
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
    
    # Write to clear sky file
    lines[iline_ctrl] = ctrl_line_clr
    lines_clr = "".join(lines)
    f_out_clr.write(lines_clr)
    f_out_clr.close()
    
    # Write to cloudy sky file
    lines[iline_ctrl] = ctrl_line_cld
    lines_cld = "".join(lines)
    f_out_cld.write(lines_cld)
    f_out_cld.close()
    
    return file_sw_out+'_clr', file_sw_out+'_cld'

# Edit the cloud input file
def edit_cld_input( file_cld_in, file_cld_out, iwp_rrtm, reff_rrtm, cldfr, cliqwp, cicewp ):
    # iwp_rrtm[:]              CONTRAIL ice water path in each grid cell on the RRTM levels
    # reff_rrtm                CONTRAIL effective radius
    # cldfr                    met data cloud fraction
    # cliqwp                   met data liquid water path
    # cicewp                   met data ice water path
    
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
    
    # Assume no cloud in the met data initially
    any_cloud = False
    
    # Record C1.2 (clear)
    iline = 1
    # for layer, cur_cldfr in enumerate(cldfr):
    for layer, cur_coniwp in enumerate(iwp_rrtm):
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
            any_cloud = True
    
    # Record C1.2 (cloudy)
    iline = 1
    # for layer, cur_cldfr in enumerate(cldfr):
    # Convert from kg/m2 to g/m2
    for layer, cur_coniwp in enumerate(iwp_rrtm*1000.0):
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
    
    return file_cld_out+'_clr', file_cld_out+'_cld', any_cloud

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

def calc_sza(lat,lon,curr_dt):
    return 90.0 - pys.get_altitude_fast(lat,lon,curr_dt.replace(tzinfo=dt.timezone.utc))

def APCEMM_RF(ts_dir,z_flight,flight_datetime,lat_vec,lon_vec,
              sw_bin,lw_bin,ref_dir,
              altitude_edges,fn_z_to_p,temperature_array,rh_array,
              dt=None,dt_max=None,
              number2sum=16,approach=0,
              min_icemass=1.0e-5,verbose=False,
              clean_dir=True,use_mca_lw=True,
              use_mca_sw=True,max_sza=90.0,
              use_single=False,use_libRadtran=False,
              dt_base=None):
             
    from run_RRTM import run_directory
    from time import time
    from datetime import timedelta
    
    if use_libRadtran:
        assert libRadtran_present, 'libRadtran is not available'
    elif (not use_mca_lw) or (not use_mca_sw):
        print('WARNING: contrail impacts are only calculated when MCA is enabled')
    
    # Step 1: Generate RRTM input files
    if dt is None:
        dt = timedelta(minutes=10)
    if dt_max is None:
        dt_max = timedelta(hours=24)
    if dt_base is None:
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

    if clean_dir:
        clean_rrtm_dir(os.path.join(ts_dir,'rrtm'))
    
    sza_vec = []
    x_data = {}
    xb_data = {}
    if use_libRadtran:
        lrt_input = {}
    t_start = time()
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
        skip_sw = np.abs(sza) > max_sza
        ncol, dx, x, x_b, y, y_b, aux = APCEMM2RRTM_V2(f_APCEMM,z_flight,
                                                       flight_datetime,number2sum,
                                                       approach,altitude_edges,fn_z_to_p,
                                                       temperature,rh,ref_dir=ref_dir_abs,
                                                       emissivity=emissivity,albnirdf=albnirdf,
                                                       albnirdr=albnirdr,albvisdf=albvisdf,
                                                       albvisdr=albvisdr,sza=sza,
                                                       verbose=False,use_mca_sw=use_mca_sw,
                                                       use_mca_lw=use_mca_lw,skip_sw=skip_sw,
                                                       use_libRadtran=use_libRadtran)
        dx_data[tstamp] = dx
        x_data[tstamp] = x
        xb_data[tstamp] = x_b
        ncol_data[tstamp] = ncol
        if use_libRadtran:
            lrt_input[tstamp] = aux # Vector of dicts - one entry per column
        dt_curr += dt
    t_stop = time()
    t_input_gen = t_stop-t_start
        
    # Step 2: Run radiative transfer
    t_start = time()
    if use_libRadtran:
        lrt_output = run_LRT_set(lrt_input)
    else:
        rrtm_dir = os.path.join(ts_dir,'rrtm')
        run_directory(rrtm_dir,sw_bin=sw_bin,lw_bin=lw_bin,verbose=verbose,use_single=use_single)
    t_stop = time()
    t_RT = t_stop-t_start
    if verbose:
        print('Completed calculations in {:.1f} seconds'.format(t_RT))

    # Step 3: Calculate forcing 
    if not use_libRadtran:
        f_list = [x for x in os.listdir(rrtm_dir) if x.startswith('lw_output_t') and x.endswith('_clr')]
    rf = {'net': [], 'sw': [], 'lw': []}
    rf_2D = {'net': [], 'sw': [], 'lw': [], 'width': []}
    dt_curr = dt_base
    t = []
    t_start = time()
    while dt_curr < dt_max:
        net = 0
        lw = 0
        sw = 0
        total_sec = dt_curr.total_seconds()
        hh = int(np.floor(total_sec/3600.0))
        mm = int(np.mod(total_sec/60.0,60))
        tstamp = '{:02d}{:02d}'.format(hh,mm)
        if use_libRadtran:
            if tstamp in lrt_input.keys():
                input_set = lrt_input[tstamp]
                output_set = lrt_output[tstamp]
                ncol = len(input_set)
            else:
                ncol = 0
        else:
            f_list_mini = [x for x in f_list if 't' + tstamp in x]
            ncol = len(f_list_mini)
        if ncol == 0:
            break
        col_width = dx_data[tstamp]
        rf_2D['width'].append(col_width)
        rf_2D['net'].append(np.zeros(ncol))
        rf_2D['lw'].append(np.zeros(ncol))
        rf_2D['sw'].append(np.zeros(ncol))
        #for icol, f in enumerate(f_list_mini):
        for icol in range(ncol):
            if use_libRadtran:
                # Convention: LW and SW are both calculated as upwelling minus downwelling
                # The output stored is the change in net outbound resulting from
                # inclusion of the contrail layer - so a positive value means
                # a negative radiative forcing
                # Change this so that it matches the RRTMG convention, where positive
                # LW values mean warming and positive SW values mean cooling
                net_col = -1.0 * output_set['net'][icol]
                lw_col = -1.0 * output_set['lw'][icol]
                sw_col = output_set['sw'][icol]
            else:
                # Convention: net = LW - SW
                # LW is positive inbound, SW is positive outbound
                # Net is taken as LW - SW
                f = f_list_mini[icol]
                column = int(f.split('_')[3][1:])
                f_lw_clr = 'lw_output_t{:s}_c{:03d}_clr'.format(tstamp,column)
                f_sw_clr = 'sw_output_t{:s}_c{:03d}_clr'.format(tstamp,column)
                f_lw_cld = 'lw_output_t{:s}_c{:03d}_cld'.format(tstamp,column)
                f_sw_cld = 'sw_output_t{:s}_c{:03d}_cld'.format(tstamp,column)
                # Net = LW - SW
                # LW/SW = Clear - Cloudy
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
    t_stop = time()
    t_postprocess = t_stop - t_start
        
    aux_data = {'rf_2D': rf_2D, 'sza': sza_vec, 'x_b': xb_data, 'x': x_data,
                'timing': {'Input': t_input_gen, 'RT': t_RT, 'Postprocessing': t_postprocess}}
    return t, rf, aux_data

def clean_rrtm_dir(dirpath,clean_input=True,clean_output=True):
    # Cleans an RRTM directory if extant
    if not os.path.isdir(dirpath):
        return
    if clean_input:
        f_list = [x for x in os.listdir(dirpath) if '_input_' in x]
        for f in f_list:
            os.remove(os.path.join(dirpath,f))
    if clean_input:
        f_list = [x for x in os.listdir(dirpath) if '_output_' in x]
        for f in f_list:
            os.remove(os.path.join(dirpath,f))
    return

def monotonic(x):
    pos = np.all(x[1:] >= x[:-1])
    neg = np.all(x[1:] <= x[:-1])
    return pos or neg

# libRadtran-based stuff
def setup_LRT(ice_data,emissivity,albedo,sza,atmosphere=None,lrt_data_path=None,env=None):
    if lrt_data_path is None:
        lrt_data_path = '/home/seastham/libRadtran/lrt_2.0.5/share/libRadtran/data'
    if env is None:
        env = {'LD_LIBRARY_PATH': '/home/seastham/libRadtran/gsl_2.7/lib:/home/seastham/.conda/envs/gcpy1.3/lib'}

    LIBRADTRAN_FOLDER = get_lrt_folder()
    
    spectrum_solar   = 'kato2'
    spectrum_thermal = 'fu'
    
    run_wolf = False
    xopts = {}
    
    ice_data_mod = copy.deepcopy(ice_data)
    if run_wolf:
        ice_model = 'yang'
        habit = 'rough-aggregate'
        r_min = 3.56 # for yang
        # Example conditions from Wolf et al. 2023
        albedo = 0.0
        emissivity = [1.0]
        sza = 0.0
        r_eff = np.where(ice_data['re'] > 0.0,85.0,0.0)
        ice_data_mod['re'] = [0,85]
        ice_data_mod['z'] = [10,9]
        ice_data_mod['iwc'] = [0,0.024]
        xopts['sur_temperature'] = 299.7
        # Ice cloud temperature of 219 K
        # No liquid clouds
        # Zero surface albedo
        # SZA of 0.0
        # Effective radius of 85 um
        # IWC of 0.024 g/m3
    else:
        ice_model = 'yang'
        habit = 'rough-aggregate'
        r_min = 3.56 # for yang
        r_eff = ice_data['re'].copy()
        ice_data_mod['re'] = np.where(np.logical_and(r_eff > 0.0,r_eff < r_min),r_min,r_eff)
    
    # Options common to both the solar and thermal calculation
    xopts['pseudospherical'] = ''
    xopts['data_files_path'] = lrt_data_path
    xopts['rte_solver'] = 'disort'
    xopts['pressure_out'] = 'toa'
    xopts['number_of_streams'] = '6' # May need to push this to 16
    
    solar_wl = '250 2600'
    thermal_wl = '2500 80000'
    
    # Begin setup
    ic_opts = {'ic_habit': habit}
    
    slrt = RadTran(LIBRADTRAN_FOLDER,env=env)
    slrt.options['sza'] = sza
    slrt.options['albedo'] = albedo
    slrt.options['source'] = 'solar'
    slrt.options['wavelength'] = solar_wl
    slrt.options['output_user'] = 'p edir edn eup'
    slrt.options['mol_abs_param'] = spectrum_solar
    if spectrum_solar in ['fu','kato2']:
        slrt.options['output_process'] = 'sum'
    else:
        slrt.options['output_process'] = 'integrate'
    slrt.atm_profile = atmosphere
        
    #if liquid_cloud:
    #    slrt.cloud = cloud_data 

    for key, val in xopts.items():
        slrt.options[key] = val
        
    # Set up the basic thermal radiation solver
    tlrt = RadTran(LIBRADTRAN_FOLDER,env=env)
    tlrt.options['source'] = 'thermal'
    tlrt.options['albedo'] = 1.0 - emissivity[0]
    tlrt.options['data_files_path'] = lrt_data_path
    tlrt.options['output_user'] = 'p edir edn eup'
    tlrt.options['wavelength'] = thermal_wl
    tlrt.options['mol_abs_param'] = spectrum_thermal
    if spectrum_thermal in ['fu','kato2']:
        tlrt.options['output_process'] = 'sum'
    else:
        tlrt.options['output_process'] = 'integrate'
    for key, val in xopts.items():
        tlrt.options[key] = val
    tlrt.atm_profile = atmosphere

    #if liquid_cloud:
    #    tlrt.cloud = cloud_data 

    # Now copy the "non-contrail" versions and add in the contrail ice layer
    slrt_cld = copy.deepcopy(slrt)
    slrt_cld.icecloud = ice_data_mod
    slrt_cld.options['ic_properties'] = ice_model
    for key, val in ic_opts.items():
        slrt_cld.options[key] = val

    tlrt_cld = copy.deepcopy(tlrt)
    tlrt_cld.icecloud = ice_data_mod
    tlrt_cld.options['ic_properties'] = ice_model
    for key, val in ic_opts.items():
        tlrt_cld.options[key] = val

    return slrt, slrt_cld, tlrt, tlrt_cld

def run_LRT_set(lrt_input,max_sza=90.0,run_test=False):
    lrt_output = {}
    for tstamp, col_set in lrt_input.items():
        ncol = len(col_set)
        lw_vec  = np.zeros(ncol)
        sw_vec  = np.zeros(ncol)
        net_vec = np.zeros(ncol)
        for icol in range(ncol):
            input_set = col_set[icol]
            if run_test:
                print(icol)
                sw_con  = input_set['slrt_cloudy'].run(verbose=False,quiet=False,print_input=True,debug=True)
                lw_con  = input_set['tlrt_cloudy'].run(verbose=False,quiet=False,print_input=True)
            else:
                sw_con  = input_set['slrt_cloudy'].run(verbose=False,quiet=True)
                lw_con  = input_set['tlrt_cloudy'].run(verbose=False,quiet=True)
            sw_base = input_set['slrt_clear'].run(verbose=False,quiet=True)
            lw_base = input_set['tlrt_clear'].run(verbose=False,quiet=True)
            sza = input_set['sza']
            # Change in NET OUTGOING shortwave and longwave radiation
            # Positive means cooling!
            # Outputs:
            # 0 -> p      Pressure
            # 1 -> edir   
            # 2 -> edn    Downwelling flux
            # 3 -> eup    Upwelling flux
            # Up - Down = Net outbound TOA flux
            # Difference between CON and BASE is the change in outbound flux
            # due to the inclusion of the countrail layer
            lw_change = (lw_con[3] - lw_con[2]) - (lw_base[3] - lw_base[2])
            # Results may not be valid at high SZA
            if sza >= max_sza:
                sw_change = 0.0
            else:
                sw_change = (sw_con[3] - sw_con[2]) - (sw_base[3] - sw_base[2])
            lw_vec[icol] = lw_change
            sw_vec[icol] = sw_change
            net_vec[icol] = lw_change + sw_change
        lrt_output[tstamp] = {'lw': lw_vec, 'sw': sw_vec, 'net': net_vec}
    if run_test:
        raise ValueError('TEST COMPLETE')
    return lrt_output