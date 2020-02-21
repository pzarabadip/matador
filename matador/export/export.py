# coding: utf-8
# Distributed under the terms of the MIT License.

""" This file implements functions that write files from matador
documents or Crystal objects.
"""

import os
from traceback import print_exc

import numpy as np

from matador.utils.cell_utils import cart2abcstar, frac2cart, cart2abc
from matador.utils.cell_utils import abc2cart, calc_mp_grid
from matador.utils.cursor_utils import display_results
from matador.utils.chem_utils import get_formula_from_stoich, get_stoich, get_root_source

EPS = 1e-8


def file_writer_function(function):
    """ Wrapper for file writers to safely overwrite/hash duplicate files. """

    from functools import wraps

    @wraps(function)
    def wrapped_writer(*args, **kwargs):
        """ Wrap and return the writer function. """
        path = args[1]
        try:
            flines, ext = function(*args, **kwargs)
            if ext is not None and not path.endswith(ext):
                path += '.{}'.format(ext)

            if os.path.isfile(path):
                if kwargs.get('overwrite'):
                    os.remove(path)
                elif kwargs.get('hash_dupe'):
                    print('File name already exists, generating hash...')
                    path = '{}-{}.{}'.format(path.replace(ext, ''), generate_hash(), ext)
                else:
                    print('File name already exists! Skipping!')
                    raise RuntimeError('Duplicate file!')

            with open(path, 'w') as f:
                for line in flines:
                    f.write(line + '\n')

        except Exception as exc:
            print_exc()
            raise type(exc)('Failed to write {}: {}'.format(path, exc))

    return wrapped_writer


def query2files(cursor, **kwargs):
    """ Many-to-many convenience function for many structures being written to
    many file types.

    Parameters:
        cursor (:obj:`list` of :obj:`dict`): list of matador dictionaries to write out.

    Keyword arguments:
        **kwargs (dict): dictionary of {filetype: bool(whether to write)}. Accepted file types
            are cell, param, res, pdb, json, xsf, markdown and latex.

    """
    cell = kwargs.get('cell')
    top = kwargs.get('top')
    param = kwargs.get('param')
    res = kwargs.get('res')
    pdb = kwargs.get('pdb')
    json = kwargs.get('json')
    xsf = kwargs.get('xsf')
    md = kwargs.get('markdown')
    tex = kwargs.get('latex')
    argstr = kwargs.get('argstr')
    multiple_files = cell or param or res or pdb or xsf
    prefix = (kwargs.get('prefix') + '-') if kwargs.get('prefix') is not None else ''
    pressure = kwargs.get('write_pressure')
    if kwargs.get('subcmd') in ['polish', 'swaps']:
        info = False
        hash_dupe = False
    else:
        info = True
        hash_dupe = False

    if isinstance(cursor, list):
        num = len(cursor)
    else:
        num = cursor.count()

    if top is not None:
        if top < num:
            cursor = cursor[:top]
            num = top
    if multiple_files:
        print('Intending to write', num, 'structures to file...')
        if len(cursor) > 10000:
            write = input('This operation will write ' + str(len(cursor)) + ' structures' +
                          ' are you sure you want to do this? [y/n] ')
            if write.lower() == 'y':
                print('Writing them all.')
                write = True
            else:
                write = False
                return
        else:
            write = True
    dirname = generate_relevant_path(kwargs)
    _dir = False
    dir_counter = 0
    while not _dir:
        if dir_counter != 0:
            directory = dirname + str(dir_counter)
        else:
            directory = dirname
        if not os.path.isdir(directory):
            os.makedirs(directory)
            _dir = True
        else:
            dir_counter += 1
    for _, doc in enumerate(cursor):
        name = prefix
        path = directory + '/'
        # write either cell, res or both
        root_source = get_root_source(doc)
        formula = get_formula_from_stoich(doc['stoichiometry'])

        if kwargs.get('subcmd') == 'swaps':
            root_source = root_source.replace('-swap-', '-')
        name = root_source

        if 'OQMD ' in root_source:
            name = '{formula}-OQMD_{src}'.format(formula=formula, src=root_source.split(' ')[-1])
        elif 'mp-' in root_source:
            name = '{formula}-MP_{src}'.format(formula=formula, src=root_source.split('-')[-1])
        if 'icsd' in doc and 'CollCode' not in name:
            name += '-CollCode{}'.format(doc['icsd'])
        else:
            pf_id = None
            for source in doc['source']:
                if 'pf-' in source:
                    pf_id = source.split('-')[-1]
                    break
            else:
                if 'pf_ids' in doc:
                    pf_id = doc['pf_ids'][0]
            if pf_id is not None:
                name += '-PF-{}'.format(pf_id)

        # if swaps, prepend new composition
        if kwargs.get('subcmd') == 'swaps':
            new_formula = get_formula_from_stoich(get_stoich(doc['atom_types']))
            name = '{}-swap-{}'.format(new_formula, name)

        path += name
        if param:
            doc2param(doc, path, hash_dupe=hash_dupe)
        if cell:
            doc2cell(doc, path, pressure, hash_dupe=hash_dupe)
        if res:
            doc2res(doc, path, info=info, hash_dupe=hash_dupe)
        if json:
            doc2json(doc, path, hash_dupe=hash_dupe)
        if pdb:
            doc2pdb(doc, path, hash_dupe=hash_dupe)
        if xsf:
            doc2xsf(doc, path)

    hull = kwargs.get('subcmd') in ['hull', 'voltage']
    md_path = path.split('/')[0] + '/' + path.split('/')[0] + '.md'
    print('Writing markdown file', md_path + '...')
    md_string = display_results(cursor, args=kwargs, argstr=argstr, markdown=True, hull=hull)
    with open(md_path, 'w') as f:
        f.write(md_string)
    if tex:
        tex_path = path.split('/')[0] + '/' + path.split('/')[0] + '.tex'
        print('Writing LaTeX file', tex_path + '...')
        tex_string = display_results(cursor, args=kwargs, argstr=argstr, latex=True, hull=hull)
        with open(tex_path, 'w') as f:
            f.write(tex_string)

    print('Done!')


@file_writer_function
def doc2param(doc, *args, **kwargs):
    """ Write basic .param file from single doc.

    Parameters:
        doc (dict): the document to write out to file
        path (str): the desired path to file

    Keyword arguments:
        spin (bool): enforce breaking of spin symmetry to magic number of 5.
        hash_dupe (bool): hash duplicate file names, or skip?
        overwrite (bool): overwrite if filename exists.

    Returns:
        list, str: list of strings to write to file, with implicit newlines,
            and the required file extension.

    """
    from matador.utils.castep_params import CASTEP_PARAMS, CASTEP_VERSION
    param_set = set(CASTEP_PARAMS)
    param_dict = dict()
    for param in [param for param in param_set if param in doc]:
        param_dict[param] = doc[param]

    flines = []
    flines.append('# Param file generated by matador (Matthew Evans 2016)\n# CASTEP version {} parameter set'
                  .format(CASTEP_VERSION))

    spin = kwargs.get('spin')
    if isinstance(spin, bool):
        if spin:
            spin = 5
        else:
            spin = None

    skip_keywords = ['nbands', 'nelectrons']
    for kw in skip_keywords:
        if kw in doc:
            print('Skipping keyword {} as it was probably not desired...'.format(kw))
            param_dict.pop(kw)

    if spin is not None and not doc.get('spin_polarized'):
        param_dict['spin_polarized'] = True

    if param_dict.get('spin_polarized'):
        total_spin = None
        if 'atomic_init_spins' in doc:
            total_spin = sum([val for val in doc['atomic_init_spins'] if val])
        elif spin is not None:
            total_spin = spin

        if total_spin is not None:
            flines.append("{0:30}: {1}".format('spin', total_spin))

    for param in param_dict:
        if param not in ['source', 'devel_code']:
            if param in ['basis_precision'] and 'cut_off_energy' in param_dict:
                continue
            flines.append("{0:30}: {1}".format(param, param_dict[param]))

    if 'encapsulated' in doc and doc['encapsulated']:
        try:
            from implicit_cnts import implicit_cnt_params
            cnt_params = implicit_cnt_params(doc['cnt_radius'])
            flines.append('%BLOCK DEVEL_CODE')
            flines.append('ADD_EXT_LOCPOT: \"gaussian_cylinder\"')
            flines.append('1D_STRESS_TENSOR: True')
            flines.append('gaussian_cylinder_pot:')
            flines.append('V0 = {V0}\nradius = {radius}\nbroadening = {fwhm}\naxial = \"0 0 1\"\ncentre = \"0.5 0.5 0\"\n'
                          .format(**cnt_params))
            flines.append(':endgaussian_cylinder_pot')
            flines.append('%ENDBLOCK DEVEL_CODE')
        except ImportError:
            print('Failed to import implicit_cnt_params from pyairss, so not writing '
                  'implicit cnt parameters from .res file to .param file.')

    if 'devel_code' in param_dict:
        flines.append('\n%BLOCK DEVEL_CODE')
        flines.append(param_dict['devel_code'].strip())
        flines.append('%ENDBLOCK DEVEL_CODE')

    ext = 'param'

    return flines, ext


@file_writer_function
def doc2cell(doc, path, *args, **kwargs):
    """ Write .cell file for single doc.

    Parameters:
        doc (dict): the document to write to file
        path (str): the desired path to the file

    Keyword Arguments:
        hash_dupe (bool): hash duplicate file names or skip?
        overwrite (bool): overwrite if filename already exists
        spin (bool): break spin symmetry with magic number 5 on first atom

    Returns:
        list, str: list of strings to write to file, with implicit newlines,
            and the required file extension.

    """
    spin = kwargs.get('spin')
    if isinstance(spin, bool):
        if spin:
            spin = 5
        else:
            spin = None

    # if spin symmetry breaking is requested, update atomic init spins
    if 'positions_frac' in doc and len(doc['positions_frac']) > 0:
        if spin or 'atomic_init_spins' in doc:
            if not any(doc.get('atomic_init_spins', [])):
                doc['atomic_init_spins'] = len(doc['positions_frac']) * [None]
                doc['atomic_init_spins'][0] = spin

    if 'atomic_init_spins' in doc and len(doc['atomic_init_spins']) != len(doc['positions_frac']):
        print('Length mismatch between atoms and spins, will pad with zeros...')

    if 'site_occupancy' in doc:
        for occ in doc['site_occupancy']:
            if abs(occ - 1) > EPS:
                raise RuntimeError('Partial occupancies (VCA) not supported in cell files with matador')

    flines = []
    flines.append('# Cell file generated by matador (Matthew Evans 2016)\n')

    if 'lattice_cart' in doc:
        flines.append('%BLOCK LATTICE_CART')
        for vec in doc['lattice_cart']:
            flines.append('{d[0]} {d[1]} {d[2]}'.format(d=vec))
        flines.append('%ENDBLOCK LATTICE_CART')

    if 'atom_types' and 'positions_frac' in doc:
        flines.append('\n%BLOCK POSITIONS_FRAC')
        for ind, atom in enumerate(zip(doc['atom_types'], doc['positions_frac'])):
            postfix = ''
            try:
                if 'atomic_init_spins' in doc and doc['atomic_init_spins'][ind]:
                    postfix = "SPIN={}".format(doc['atomic_init_spins'][ind])
            except IndexError:
                pass

            flines.append("{0:8s} {1[0]: 15f} {1[1]: 15f} {1[2]: 15f} {2:}".format(
                atom[0], atom[1], postfix))
        flines.append('%ENDBLOCK POSITIONS_FRAC')

    if 'external_pressure' in doc:
        if np.asarray(doc['external_pressure']).shape != (3, 3):
            raise RuntimeError('External pressure has wrong shape: {}'.format(doc['external_pressure']))
        flines.append('\n%BLOCK EXTERNAL_PRESSURE')
        flines.append('{d[0][0]} {d[0][1]} {d[0][2]}'.format(d=doc['external_pressure']))
        flines.append('{d[1][1]} {d[1][2]}'.format(d=doc['external_pressure']))
        flines.append('{d[2][2]}'.format(d=doc['external_pressure']))
        flines.append('%ENDBLOCK EXTERNAL_PRESSURE')

    if 'ionic_constraints' in doc:
        flines.append('\n%BLOCK IONIC_CONSTRAINTS')
        for constraint in doc['ionic_constraints']:
            flines.append('{}'.format(constraint))
        flines.append('%ENDBLOCK IONIC_CONSTRAINTS')

    # specify SCF kpoints
    if 'kpoints_list' in doc:
        flines.append('\n%BLOCK KPOINTS_LIST')
        for kpoint in doc['kpoints_list']:
            flines.append('{p[0]:f} {p[1]:f} {p[2]:f} {p[3]:f}'.format(p=kpoint))
        flines.append('\n%ENDBLOCK KPOINTS_LIST')
    elif 'kpoints_mp_spacing' in doc:
        flines.append('\nKPOINTS_MP_SPACING: {}'.format(doc['kpoints_mp_spacing']))
    elif 'kpoints_mp_grid' in doc:
        flines.append('\nKPOINTS_MP_GRID: {d[0]} {d[1]} {d[2]}'.format(d=doc['kpoints_mp_grid']))

    if 'supercell_kpoints_mp_spacing' in doc:
        flines.append('\nSUPERCELL_KPOINTS_MP_SPACING: {}'.format(doc['supercell_kpoints_mp_spacing']))

    if 'kpoints_mp_offset' in doc:
        flines.append('\nKPOINTS_MP_OFFSET: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['kpoints_mp_offset']))

    if 'spectral_kpoints_mp_offset' in doc:
        flines.append('SPECTRAL_KPOINTS_MP_OFFSET: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['spectral_kpoints_mp_offset']))
    if 'phonon_kpoint_mp_offset' in doc:
        flines.append('PHONON_KPOINT_MP_OFFSET: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['phonon_kpoint_mp_offset']))

    # specify spectral kpoints
    if 'spectral_kpoints_mp_spacing' in doc:
        flines.append('\nSPECTRAL_KPOINTS_MP_SPACING: {}'
                      .format(doc['spectral_kpoints_mp_spacing']))
    elif 'spectral_kpoints_mp_grid' in doc:
        flines.append('\nSPECTRAL_KPOINTS_MP_GRID: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['spectral_kpoints_mp_grid']))
    elif 'spectral_kpoints_list' in doc:
        flines.append('\n%BLOCK SPECTRAL_KPOINTS_LIST')
        for point in doc['spectral_kpoints_list']:
            flines.append('{p[0]} {p[1]} {p[2]}'.format(p=point))
        flines.append('%ENDBLOCK SPECTRAL_KPOINTS_LIST')
    elif 'spectral_kpoints_path' in doc:
        flines.append('\n%BLOCK SPECTRAL_KPOINTS_PATH')
        for ind, point in enumerate(doc['spectral_kpoints_path']):
            flines.append('{p[0]} {p[1]} {p[2]}'.format(p=point))
            if 'spectral_kpoints_path_labels' in doc:
                flines[-1] += ' ! {}'.format(doc['spectral_kpoints_path_labels'][ind])
        flines.append('%ENDBLOCK SPECTRAL_KPOINTS_PATH')
    if 'spectral_kpoints_path_spacing' in doc:
        flines.append('\nSPECTRAL_KPOINTS_PATH_SPACING: {}'.format(doc['spectral_kpoints_path_spacing']))

    # specify phonon kpoints
    if 'phonon_kpoint_list' in doc:
        flines.append('\n%BLOCK PHONON_KPOINT_LIST')
        for point in doc['phonon_kpoint_list']:
            flines.append('{p[0]} {p[1]} {p[2]}'.format(p=point))
        flines.append('%ENDBLOCK PHONON_KPOINT_LIST')
    elif 'phonon_kpoint_mp_spacing' in doc:
        flines.append('\nPHONON_KPOINT_MP_SPACING: {}'
                      .format(doc['phonon_kpoint_mp_spacing']))
    elif 'phonon_kpoint_mp_grid' in doc:
        flines.append('\nPHONON_KPOINT_MP_GRID: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['phonon_kpoint_mp_grid']))

    # specify phonon fine kpoints
    if 'phonon_fine_kpoint_list' in doc:
        flines.append('\n%BLOCK PHONON_FINE_KPOINT_LIST')
        for point in doc['phonon_fine_kpoint_list']:
            flines.append('{p[0]} {p[1]} {p[2]}'.format(p=point))
        flines.append('%ENDBLOCK PHONON_FINE_KPOINT_LIST')
    elif 'phonon_fine_kpoint_mp_spacing' in doc:
        flines.append('\nPHONON_FINE_KPOINT_MP_SPACING: {}'
                      .format(doc['phonon_fine_kpoint_mp_spacing']))
    elif 'phonon_fine_kpoint_mp_grid' in doc:
        flines.append('\nPHONON_FINE_KPOINT_MP_GRID: {d[0]} {d[1]} {d[2]}'
                      .format(d=doc['phonon_fine_kpoint_mp_grid']))
    elif 'phonon_fine_kpoint_path' in doc:
        flines.append('\n%BLOCK PHONON_FINE_KPOINT_PATH')
        for ind, point in enumerate(doc['phonon_fine_kpoint_path']):
            flines.append('{p[0]} {p[1]} {p[2]}'.format(p=point))
            if 'phonon_fine_kpoint_path_labels' in doc:
                flines[-1] += ' ! {}'.format(doc['phonon_fine_kpoint_path_labels'][ind])
        flines.append('%ENDBLOCK PHONON_FINE_KPOINT_PATH')
    if 'phonon_fine_kpoint_path_spacing' in doc:
        flines.append('\nPHONON_FINE_KPOINT_PATH_SPACING: {}'.format(doc['phonon_fine_kpoint_path_spacing']))

    if 'phonon_supercell_matrix' in doc:
        flines.append('\n%BLOCK PHONON_SUPERCELL_MATRIX')
        for i in range(3):
            flines.append('{d[0]:3d} {d[1]:3d} {d[2]:3d}'.format(d=doc['phonon_supercell_matrix'][i]))
        flines.append('%ENDBLOCK PHONON_SUPERCELL_MATRIX')

    if 'cell_constraints' in doc:
        flines.append('\n%BLOCK CELL_CONSTRAINTS')
        flines.append('{d[0]} {d[1]} {d[2]}'.format(d=doc['cell_constraints'][0]))
        flines.append('{d[0]} {d[1]} {d[2]}'.format(d=doc['cell_constraints'][1]))
        flines.append('%ENDBLOCK CELL_CONSTRAINTS')

    if doc.get('fix_com'):
        flines.append('FIX_COM: TRUE')
    if doc.get('fix_all_cell'):
        flines.append('FIX_ALL_CELL: TRUE')
    if doc.get('fix_all_ions'):
        flines.append('FIX_ALL_IONS: TRUE')
    if doc.get('fix_vol'):
        flines.append('FIX_VOL: TRUE')

    if 'symmetry_generate' in doc:
        flines.append('SYMMETRY_GENERATE')
    if 'snap_to_symmetry' in doc:
        flines.append('SNAP_TO_SYMMETRY')
    if 'symmetry_tol' in doc:
        flines.append('SYMMETRY_TOL: {}'.format(doc['symmetry_tol']))
    if 'hubbard_u' in doc:
        flines.append('\n%BLOCK HUBBARD_U')
        flines.append('eV')
        for elem in doc['hubbard_u']:
            for orbital in doc['hubbard_u'][elem]:
                shift = str(doc['hubbard_u'][elem][orbital])
                flines.append('{} {} {}'.format(elem, orbital, shift))
        flines.append('%ENDBLOCK HUBBARD_U')
    if 'quantisation_axis' in doc:
        flines.append('QUANTISATION_AXIS: ')
        for integer in doc['quantisation_axis']:
            flines.append(str(integer) + ' ')
    if 'positions_noise' in doc:
        flines.append('POSITIONS_NOISE: {}'.format(doc['positions_noise']))
    if 'cell_noise' in doc:
        flines.append('CELL_NOISE : {}'.format(doc['cell_noise']))
    if 'species_pot' in doc:
        flines.append('\n%BLOCK SPECIES_POT')
        for elem in doc['species_pot']:
            if elem == 'library':
                flines.append(doc['species_pot']['library'])
            else:
                flines.append('{:4} {}'.format(elem, doc['species_pot'][elem]))
        flines.append('%ENDBLOCK SPECIES_POT')

    ext = 'cell'

    return flines, ext


def doc2pdb(doc, path, info=True, hash_dupe=True):
    """ Write a simple .pdb for single doc.

    Parameters:
        doc (dict): matador document containing structure.
        path (str): desired path of xsf file.

    Keyword arguments:
        info (bool): write info string to HEADER.
        hash_dupe (bool): hash duplicate file names or skip?

    """
    if path.endswith('.pdb'):
        path = path.replace('.pdb', '')
    try:
        if os.path.isfile(path+'.pdb'):
            if hash_dupe:
                print('File already exists, generating hash...')
                path += '-' + generate_hash()
            else:
                raise RuntimeError('Skipping duplicate structure...')
        with open(path+'.pdb', 'w') as f:
            try:
                header = 'HEADER    {} {}'.format(doc['text_id'][0], doc['text_id'][1])
            except Exception:
                header = 'HEADER    Generated with matador.'
            try:
                # write res file header if info
                title = 'TITLE     '
                title += path.split('/')[-1] + ' '
                if not doc.get('pressure'):
                    title += '0.00 '
                else:
                    title += str(doc['pressure']) + ' '
                title += str(doc['cell_volume']) + ' '
                title += str(doc['enthalpy']) + ' '
                title += '0 0 '             # spin
                title += str(doc['num_atoms']) + ' '
                try:
                    if 'x' in doc['space_group']:
                        title += '(P1) '
                    else:
                        title += '(' + str(doc['space_group']) + ')' + ' '
                except Exception:
                    title += '(P1) '
                title += 'n - 1'
            except Exception:
                if not info:
                    title = 'TITLE\t' + path.split('/')[-1]
                raise RuntimeError('Failed to get info for res file, turn info off.')
            author = 'AUTHOR    Generated with matador (Matthew Evans, 2016)'
            f.write(header + '\n')
            f.write(author + '\n')
            f.write(title + '\n')
            # use dummy SG for CRYST1, shouldn't matter
            cryst = ('CRYST1 {v[0][0]:9.3f} {v[0][1]:9.3f} {v[0][2]:9.3f} {v[1][0]:7.2f} {v[1][1]:7.2f} {v[1][2]:7.2f} P 1'
                     .format(v=doc['lattice_abc']))
            f.write(cryst + '\n')
            scale_n = cart2abcstar(doc['lattice_cart'])
            f.write('SCALE1    {v[0][0]:10.6f} {v[0][1]:10.6f} {v[0][2]:10.6f}      {:10.5f}\n'.format(0.0, v=scale_n))
            f.write('SCALE2    {v[1][0]:10.6f} {v[1][1]:10.6f} {v[1][2]:10.6f}      {:10.5f}\n'.format(0.0, v=scale_n))
            f.write('SCALE3    {v[2][0]:10.6f} {v[2][1]:10.6f} {v[2][2]:10.6f}      {:10.5f}\n'.format(0.0, v=scale_n))
            if 'positions_abs' not in doc:
                doc['positions_abs'] = frac2cart(doc['lattice_cart'], doc['positions_frac'])
            for ind, atom in enumerate(doc['atom_types']):
                try:
                    hetatm = 'HETATM '
                    # append 00 to atom type, a la cell2pdb...
                    hetatm += '{:4d} {:.4} NON A   1     '.format(ind+1, atom+'00')
                    hetatm += ('{v[0]:7.3f} {v[1]:7.3f} {v[2]:7.3f} {:5.2f} {:5.2f}          {:.2}'
                               .format(1.0, 0.0, atom, v=doc['positions_abs'][ind]))
                    f.write(hetatm + '\n')
                except Exception:
                    print_exc()
            ter = 'TER       {}       NON A   1'.format(len(doc['atom_types']))
            f.write(ter + '\n')
            f.write('END')
    except Exception:
        if hash_dupe:
            print_exc()
            print('Writing pdb file failed for ', doc['text_id'])
        else:
            print_exc()


def doc2json(doc, path, overwrite=False, hash_dupe=True):
    """ Return raw JSON document as stored in database.

    Parameters:
        doc (dict): matador document containing structure
        path (str): desired filename for res file

    Keyword arguments:
        hash_dupe (bool): hash duplicate file names, or skip?
        overwrite (bool): overwrite if filename exists.

    """
    try:
        import json
        if path.endswith('.json'):
            path = path.replace('.json', '')

        if os.path.isfile(path+'.json'):
            if overwrite:
                os.remove(path + '.json')
            elif hash_dupe:
                print('File name already exists, generating hash...')
                path += '-' + generate_hash()
            else:
                print('File name already exists! Skipping!')

        if '_id' in doc:
            doc['_id'] = str(doc['_id'])

        with open(path + '.json', 'w') as f:
            f.write(json.dumps(doc, skipkeys=True, indent=2))
    except Exception:
        print_exc()
        print('Writing {}.json failed!'.format(path))


def doc2pwscf(doc, path, template=None, spacing=None):
    """ Write the structural part of QE input file based
    on the provided matador doc. Will calculate the correct
    kpoint_mp_grid if spacing is provided.

    Parameters:
        doc (dict): matador document containing structure
        path (str): desired filename for res file

    Keyword Arguments:
        template (str): filename of template to prepend before structure
            (with prefix keyword to be replaced),
        spacing (float): kpoint_mp_spacing to use when calculating grid.

    """
    if path.endswith('.in'):
        path = path.replace('.in', '')

    if os.path.isfile(path + '.in'):
        print('File already exists, not overwriting...')
        return

    if 'lattice_cart' not in doc:
        doc['lattice_cart'] = abc2cart(doc['lattice_abc'])

    if 'kpoints_mp_spacing' in doc or spacing is not None:
        if 'kpoints_mp_spacing' in doc:
            spacing = doc['kpoints_mp_spacing']
        doc['kpoints_mp_grid'] = calc_mp_grid(doc['lattice_cart'], spacing)

    print(doc['kpoints_mp_grid'])

    file_string = ''
    file_string += 'CELL_PARAMETERS angstrom\n'
    for i in range(3):
        file_string += ' {d[0]: 10.10f} {d[1]: 10.10f} {d[2]: 10.10f}\n'.format(d=doc['lattice_cart'][i])

    file_string += '\n ATOMIC_POSITIONS crystal\n'
    for i in range(len(doc['atom_types'])):
        file_string += ('{:4} {d[0]: 10.10f} {d[1]: 10.10f} {d[2]: 10.10f}\n'
                        .format(doc['atom_types'][i], d=doc['positions_frac'][i]))
    file_string += '\nK_POINTS automatic\n'
    file_string += '{d[0]} {d[1]} {d[2]} 0 0 0'.format(d=doc['kpoints_mp_grid'])

    if template is not None:
        if os.path.isfile(template):
            with open(template, 'r') as f:
                template_string = f.readlines()

    with open(path + '.in', 'w') as f:
        for line in template_string:
            if 'prefix' in line:
                line = '  prefix = \'{}\',\n'.format(path)
            elif 'nat' in line:
                line = '  nat = {},\n'.format(len(doc['atom_types']))
            elif 'nat' in line:
                line = '  ntyp = {},\n'.format(len(set(doc['atom_types'])))
            f.write(line)
        f.write(file_string)


@file_writer_function
def doc2res(doc, path, info=True, spoof_titl=False, sort_atoms=True, **kwargs):
    """ Write .res file for single doc.

    Parameters:
        doc (dict): matador document containing structure
        path (str): desired filename for res file

    Keyword Arguments:
        info (bool): require info in res file header
        hash_dupe (bool): add random hash to colliding filenames
        spoof_titl (bool): make up fake info for file header (for use with e.g. cryan)
        overwrite (bool): overwrite files with conflicting filenames
        sorted (bool): if False, atoms are not sorted (this will not be a valid res file)

    """
    if spoof_titl:
        info = False
    if not info:
        space_group = 'P1' if 'space_group' not in doc else doc['space_group']
        if spoof_titl:
            titl = ('TITL {} -1 1 -1 0 0 {} ({}) n - 1').format(path.split('/')[-1], doc['num_atoms'], space_group)
        else:
            titl = ('TITL file generated by matador (Matthew Evans 2016)')
    else:
        try:
            titl = 'TITL '
            titl += (path.split('/')[-1] + ' ')
            if 'pressure' not in doc or isinstance(doc['pressure'], str):
                titl += '0.00 '
            else:
                titl += str(doc['pressure']) + ' '
            if 'cell_volume' not in doc:
                titl += '0.0 '
            else:
                titl += str(doc['cell_volume']) + ' '
            if 'enthalpy' in doc and not isinstance(doc['enthalpy'], str):
                titl += str(doc['enthalpy']) + ' '
            elif '0K_energy' in doc:
                titl += str(doc['0K_energy']) + ' '
            elif 'total_energy' in doc:
                titl += str(doc['total_energy']) + ' '
            else:
                raise KeyError('No energy field found.')
            titl += '0 0 '             # spin
            titl += str(doc['num_atoms']) + ' '
            if 'space_group' not in doc:
                titl += '(P1) '
            elif 'x' in doc['space_group']:
                titl += '(P1) '
            else:
                titl += '(' + str(doc['space_group']) + ')' + ' '
            titl += 'n - 1'
        except Exception:
            raise RuntimeError('Failed to get info for res file, turn info off.')
    if 'encapsulated' in doc and doc['encapsulated']:
        rem = ("REM NTPROPS {{\'chiralN\': {}, \'chiralM\': {}, \'r\': {}, "
               "\'offset\': [0.5, 0.5, 0.5], \'date\': \'xxx\', \'eformperfu\': 12345, \'z\': {}}}"
               .format(doc['cnt_chiral'][0], doc['cnt_chiral'][1], doc['cnt_radius'], doc['cnt_length']))
    flines = []
    if 'encapsulated' in doc and doc['encapsulated']:
        flines.append(rem)
    flines.append(titl)
    cell_str = 'CELL 1.0 '
    if ('lattice_abc' not in doc or len(doc['lattice_abc']) != 2
            or len(doc['lattice_abc'][0]) != 3 or len(doc['lattice_abc'][1]) != 3):
        try:
            doc['lattice_abc'] = cart2abc(doc['lattice_cart'])
        except Exception:
            raise RuntimeError('Failed to get lattice, something has gone wrong for {}'.format(path))
    for vec in doc['lattice_abc']:
        for coeff in vec:
            cell_str += ('{:.12f} '.format(coeff))
    flines.append(cell_str)
    flines.append('LATT -1')

    # enforce correct order by elements, sorting only the atom_types, not the positions inside them
    if len(doc['positions_frac']) != len(doc['atom_types']):
        raise RuntimeError('Atom/position array mismatch!')

    if 'site_occupancy' in doc:
        if len(doc['site_occupancy']) != len(doc['positions_frac']):
            raise RuntimeError('Occupancy/position array mismatch!')

    if 'site_occupancy' not in doc:
        occupancies = [1.0] * len(doc['positions_frac'])

    if sort_atoms:
        positions_frac, atom_types = zip(*[(pos, types) for (types, pos) in
                                           sorted(zip(doc['atom_types'], doc['positions_frac']),
                                                  key=lambda k: k[0])])
        if 'site_occupancy' in doc:
            occupancies, _atom_types = zip(*[(occ, types) for (types, occ) in
                                             sorted(zip(doc['atom_types'], doc['site_occupancy']),
                                                    key=lambda k: k[0])])

    else:
        positions_frac = doc['positions_frac']
        atom_types = doc['atom_types']

    if len(positions_frac) != len(doc['positions_frac']) or len(atom_types) != len(doc['atom_types']):
        raise RuntimeError('Site occupancy mismatch!')

    written_atoms = []
    sfac_str = 'SFAC \t'
    for elem in atom_types:
        if elem not in written_atoms:
            sfac_str += ' ' + str(elem)
            written_atoms.append(str(elem))
    flines.append(sfac_str)
    atom_labels = []
    i = 0
    j = 1
    while i < len(atom_types):
        num = atom_types.count(atom_types[i])
        atom_labels.extend(num*[j])
        i += num
        j += 1

    for atom in zip(atom_types, atom_labels, positions_frac, occupancies):
        flines.append("{0:8s}{1:3d}{2[0]: 15f} {2[1]: 15f} {2[2]: 15f}  {3: 15f}".format(
            atom[0], atom[1], atom[2], atom[3]))
    flines.append('END')
    # very important newline for compatibliy with cryan
    # flines.append('')

    return flines, 'res'


def doc2xsf(doc, path, write_energy=False, write_forces=False, overwrite=False):
    """ Write an .xsf file for a matador document, with positions in
    Cartesian coordinates. Optionally, write the energy in a comment
    at the top of the file for use with aenet.

    Parameters:
        doc (dict): matador document containing structure.
        path (str): desired path of xsf file.

    Keyword arguments:
        write_energy (bool): whether or not to write total energy in a comment
            as the first line of the file.
        write_forces (bool): whether or not to write the forces on each atom.
        overwrite (bool): overwrite if file exists.

    """
    if path.endswith('.xsf'):
        path = path.replace('.xsf', '')

    flines = []
    if write_energy:
        if 'total_energy' in doc:
            flines.append('# total energy = {:10.8f} eV\n'.format(doc['total_energy']))
        else:
            raise RuntimeError("Failed to write energy in xsf file: key 'total_energy' missing from input.")

    flines.append('CRYSTAL')
    flines.append('PRIMVEC')
    if 'lattice_cart' in doc:
        for i in range(3):
            flines.append('\t\t{lat[0]: 10.8f}\t{lat[1]: 10.8f}\t{lat[2]: 10.8f}'.format(lat=doc['lattice_cart'][i]))
    else:
        raise RuntimeError("Failed to write lattice in xsf file: key 'lattice_cart' missing from input.")
    flines.append('PRIMCOORD')
    flines.append('{} {}'.format(doc['num_atoms'], 1))
    if 'positions_abs' not in doc:
        doc['positions_abs'] = frac2cart(doc['lattice_cart'], doc['positions_frac'])
    for ind, (atom, position) in enumerate(zip(doc['atom_types'], doc['positions_abs'])):
        flines.append('{:2}\t{pos[0]: 16.8f}\t{pos[1]: 16.8f}\t{pos[2]: 16.8f}'.format(atom, pos=position))
        if write_forces:
            if 'forces' in doc:
                flines[-1] += ('\t{f[0]: 16.8f}\t{f[1]: 16.8f}\t{f[2]: 16.8f}'.format(f=doc['forces'][ind]))
            else:
                raise RuntimeError("Failed to write forces in xsf file: key 'forces' missing from input.")

    if os.path.isfile(path + '.xsf'):
        if overwrite:
            os.remove(path + '.xsf')
        else:
            print('File name already exists! Skipping!')
            raise RuntimeError('Duplicate file!')

    with open(path + '.xsf', 'w') as f:
        for line in flines:
            f.write(line + '\n')


@file_writer_function
def doc2arbitrary(doc, *args, **kwargs):
    """ Write a Python dictionary into a standard CASTEP-style
    keyword: value file.

    Parameters:
        doc (dict): contains key-value pairs.
        path (str): filename to write to.

    Keyword arguments:
        hash_dupe (bool): hash duplicate file names, or skip?
        overwrite (bool): overwrite if filename exists.

    Returns:
        list: list of strs to write to file.

    """
    ext = None
    flines = []
    output_doc = {}
    # sanitise dict to include only unique-by-case keys
    for key in doc:
        if key.startswith('_'):
            continue
        if key.lower() == 'source':
            continue
        if key.lower() in output_doc:
            raise Warning('Key {} defined multiple times.'.format(key))
        output_doc[key.lower()] = doc[key]

    for key in output_doc:
        flines.append("{}: {}".format(key, output_doc[key]))
    return flines, ext


def generate_hash(hash_len=6):
    """ Quick hash generator, based on implementation in PyAIRSS by J. Wynn.

    Keyword arguments:
        hash_len (int): desired length of hash.

    """
    hash_chars = [str(x) for x in range(10)] + [chr(97+i) for i in range(25)]
    _hash = ''
    for _ in range(hash_len):
        _hash += np.random.choice(hash_chars)
    return _hash


def generate_relevant_path(args):
    """ Generates a suitable path name based on query. """
    dirname = ''
    if args.get('subcmd') is not None:
        dirname += args.get('subcmd') + '-'
    else:
        dirname = 'query'
    if args.get('composition') is not None:
        for comp in args['composition']:
            dirname += comp
    elif args.get('formula') is not None:
        dirname += args.get('formula')[0]
    if args.get('db') is not None:
        dirname += '-' + args.get('db')[0]
    if args.get('swap') is not None:
        for swap in args['swap']:
            dirname += '-' + swap
        if args.get('hull_cutoff') is not None:
            dirname += '-hull-' + str(args.get('hull_cutoff')) + 'eV'
    if args.get('id') is not None:
        dirname += '-' + args.get('id')[0] + '_' + args.get('id')[1]
    dirname = dirname.replace('--', '-')
    return dirname
