# I/O tools
# Author: Stefano Riva, PhD Student, NRG, Politecnico di Milano
# Latest Code Update: 07 October 2025
# Latest Doc  Update: 07 October 2025

from .functions_list import FunctionsList

import h5py

import os
import glob
import numpy as np
import fluidfoam as of
import pyvista as pv

## Routine for importing from OpenFOAM using pyvista and fluidfoam
class ReadFromOF():
    r"""
    A class used to import data from OpenFOAM.

    Either the fluidfoam library (see https://github.com/fluiddyn/fluidfoam) or pyvista are exploited for the import process
        - Supported OpenFoam Versions : 2.4.0, 4.1 to 9 (org), v1712plus to v2212plus (com) - to check
        
    Parameters
    ----------
    path : str
        Path of the OpenFOAM case directory.
    skip_zero_time : bool, optional
        If `True`, the zero time folder is skipped. Default is `False`.
        
    """
    def __init__(self, path: str,
                 skip_zero_time: bool = False) -> None:

        self.path = path

        # Check if any file with .foam extension exists in the directory
        foam_files = glob.glob(os.path.join(path, '*.foam'))
        
        if not foam_files:
            # If no .foam file exists, create foam.foam
            foam_file_path = os.path.join(path, 'foam.foam')
            with open(foam_file_path, 'w') as file:
                file.write('')
        else:
            # Use the first found .foam file
            foam_file_path = foam_files[0]
        
        self.reader = pv.POpenFOAMReader(foam_file_path)
        self.reader.skip_zero_time = skip_zero_time

    def mesh(self):
        """
        Returns the mesh of the OpenFOAM case.
        
        Returns
        -------
        mesh : pyvista.UnstructuredGrid
            The mesh of the OpenFOAM case.
        """
        grid = self.reader.read()['internalMesh']
        grid.clear_data() # remove all the data
        return grid

    def save_mesh(self, filename: str):
        """
        Saves the mesh of the OpenFOAM case to a vtk-file.
        
        Parameters
        ----------
        filename : str
            Name of the file to save the mesh.
        """
        mesh = self.mesh()
        mesh.save(filename + '.vtk')

    def import_field(self, var_name: str, 
                     use_fluidfoam: bool = False, 
                     extract_cell_data: bool = True,
                     verbose: bool = True):
        r"""
        Importing all time instances (**skipping zero folder**) from OpenFOAM directory.

        Parameters
        ----------
        var_name : str
            Name of the field to import.
        use_fluidfoam : boolean, (Default: False)
            If `True`, the fluidfoam library is used for reading the data (only cell data supported); otherwise, pyvista is used.
        vector : boolean, (Default: False)
            Labelling if the field is a vector (needed only is `use_fluidfoam==True`).
        extract_cell_data : boolean, (Default: True)
            If `True`, the cell data from centroids is extracted; otherwise, point data is extracted.
        verbose: boolean, (Default = True) 
            If `True`, printing is enabled
            
        Returns
        -------
        field : list
            Imported list of functions (each element is a `numpy.ndarray`), sorted in time.
        time_instants : list
            Sorted list of time.
        """
        
        field = list()
        time_instants = list()
        
        if use_fluidfoam and extract_cell_data:
            
            for jj, file in enumerate(os.listdir(self.path)):

                if verbose:
                    print('Importing '+var_name+f' using fluidfoam - {(jj+1)/len(os.listdir(self.path))*100:.2f}%', end="\r")
            

                if not ((file == '0.orig') or (file == '0') or (file == '0.ss')):
                    d = os.path.join(self.path, file)
                    if os.path.isdir(d):

                        if os.path.exists(d+'/'+var_name):
                            try:
                                field.append( of.readscalar(self.path, file, var_name, verbose=False).reshape(-1,1) )
                            except ValueError:
                                field.append(of.readvector(self.path, file, var_name, verbose=False).T)
                                
                            time_instants.append(float(file))
                
            # Sorting snapshots according to the params_np
            indices = sorted(range(len(time_instants)), key=lambda index: time_instants[index])
            
            tmp = field.copy()
            field.clear()

            assert len(tmp) == len(indices)
            for ii in range(len(indices)):
                field.append(tmp[indices[ii]])
            time_instants.sort()    
            
        else: 
            for idx_t in range(len(self.reader.time_values)):
                if verbose:
                    print('Importing '+var_name+f' using pyvista - {(idx_t+1)/len(self.reader.time_values)*100:.2f}%', end="\r")

                self.reader.set_active_time_value(self.reader.time_values[idx_t])

                if extract_cell_data: # extract centroids data
                    field.append(self.reader.read()['internalMesh'].cell_data[var_name])
                else: # extract vertices data
                    field.append(self.reader.read()['internalMesh'].point_data[var_name])
                time_instants.append(self.reader.time_values[idx_t])
                
        # Convert list to FunctionsList
        snaps = FunctionsList(dofs=field[0].flatten().shape[0])
        for f in field:
            snaps.append(f.flatten())

        return snaps, time_instants
    
def ImportFunctionsList(filename: str, format: str = 'h5', return_var_name: bool = False):
    """
    This function can be used to load from `xdmf/h5` files scalar or vector list of functions.

    Parameters
    ----------
    filename : str
        Name of the file to read as xdmf/h5 file.
    format : str, optional (Default = 'h5')
        Format of the file to read. It can be either 'h5', 'npy', or 'npz'.
    return_var_name : bool, optional (Default = False)
        If `True`, the variable name is returned along with the FunctionsList.
    
    Returns
    -------
    snap : FunctionsList
        Imported list of functions.
    """
    
    fmt = format.lower()
    
    if fmt == 'h5':
        with h5py.File(filename + '.h5', 'r') as f:
            var_name = list(f.keys())[0]
            data = f[var_name][:]
    elif fmt == 'npz':
        data = np.load(filename + '.npz')
        var_name = list(data.keys())[0]
        data = data[var_name]
    else:
        raise ValueError(f"Unsupported format {fmt}. Use 'h5' or 'npz'.")
    
    # Create FunctionsList from the data
    snap = FunctionsList(dofs=data.shape[0])
    snap.build_from_matrix(data)

    if return_var_name:
        return snap, var_name
    return snap
