from temp_shared_globals import refpts
import pymcx.pymcx as mcxm
import numpy as np
from fw_model import Fw_model
from probe import Probe
import nibabel as nib


def calculate_fluence(fwmodel: Fw_model, probe: Probe, volume_file):

    # Null checks
    if not fwmodel:
        raise ValueError("fwmodel cannot be None.")
    if not probe:
        raise ValueError("probe cannot be None.")
    if not volume_file:
        raise ValueError("volume_file cannot be provided as None.")

    # Check expected attributes in fwmodel and probe
    required_fwmodel_attrs = [
        'mesh_orig', 'mesh_scalp_orig', 'tiss_prop', 'headvol'
    ]
    for attr in required_fwmodel_attrs:
        if not hasattr(fwmodel, attr):
            raise AttributeError(
                f"fwmodel is missing required attribute: {attr}")

    required_probe_attrs = ['reg_optpos']
    for attr in required_probe_attrs:
        if not hasattr(probe, attr):
            raise AttributeError(
                f"probe is missing required attribute: {attr}")

    nopt = len(probe.reg_optpos)
    num_wavelengths = 2
    num_phot = 1000000
    # nNode = 19977

    nodeX = fwmodel.mesh_orig.reduced_vertices
    elem = fwmodel.mesh_orig.reduced_faces

    if nodeX is None or elem is None:
        raise ValueError("Missing required mesh data in fwmodel.mesh_orig")

    nNode = np.size(nodeX, 0)

    nodeX_s = fwmodel.mesh_scalp_orig.reduced_vertices
    elem_s = fwmodel.mesh_scalp_orig.reduced_faces
    if nodeX_s is None or elem_s is None:
        raise ValueError(
            "Missing required mesh data in fwmodel.mesh_scalp_orig")

    nNode_s = np.size(nodeX_s, 0)
    flueMesh = np.zeros((nNode, nopt, num_wavelengths))
    flueDet = np.zeros((nopt, nopt, num_wavelengths))
    maxNVoxPerNode = fwmodel.get_projVolToMesh_brain().shape[1]

    flueMesh_scalp = np.zeros((nNode_s, nopt, num_wavelengths))

    tiss_prop = fwmodel.tiss_prop
    if tiss_prop is None:
        raise ValueError("fwmodel.tiss_prop cannot be None.")

    vol_shape = list(fwmodel.headvol.vol_t_shape)
    if not vol_shape:
        raise ValueError("Invalid volume shape in fwmodel.headvol.")

    # hWait = None  # Initialize the waitbar (if applicable)
    for iWav in range(0, num_wavelengths):
        # Loop over number of optodes
        for ii in range(0, nopt):
            # if hWait is not None:
            #    waitbar_val = ((iWav - 1) * nopt + ii) / (num_wavelengths * nopt)
            #    waitbar_msg = f'Running MCXlab for optode {((iWav - 1) * nopt + ii)} of {(num_wavelengths * nopt)}'
            #    # Update the waitbar with the current progress
            #    update_waitbar(hWait, waitbar_val, waitbar_msg)

            # cfg = {}
            cfg = mcxm.create()
            cfg["Domain"]['VolumeFile'] = volume_file

            # vol_shape = [193, 285, 225]
            print('vol_shape', vol_shape)
            cfg["Domain"]["Dim"] = vol_shape

            # cfg["Forward"]['T0'] = time_gates[0][0]
            # cfg["Forward"]['T1'] = time_gates[0][1]
            # cfg["Forward"]['Dt'] = time_gates[0][2]

            cfg["Optode"]["Source"]["Pos"] = [probe.reg_optpos[ii]
                                              [0], probe.reg_optpos[ii][1], probe.reg_optpos[ii][2]]
            srcdir = [probe.reg_optpos[ii][3],
                      probe.reg_optpos[ii][4], probe.reg_optpos[ii][5]]
            cfg["Optode"]["Source"]["Dir"] = list(
                srcdir / np.linalg.norm(srcdir))

            # print(np.linalg.norm(srcdir))
            # print(list(srcdir / np.linalg.norm(srcdir)))
            # print(srcdir)

            cfg["Optode"]["Detector"] = {}

            cfg["Domain"]['OriginType'] = 1  # cfg['issrcfrom0'] = 1
            # cfg['isnormalized'] = 1mn -d5-

            # mua -absorption, mus - scattering, g - anisotropy, n - refraction

            result_list = [{"mua": 0, "mus": 0, "g": 1, "n": 1}]

            result_list.extend([{
                "mua": tiss_prop[i]['absorption'],  # [iWav],
                "mus": tiss_prop[i]['scattering'],  # [iWav],
                "g": tiss_prop[i]['anisotropy'],  # [0],
                "n": tiss_prop[i]['refraction']  # [0]
            } for i in range(len(tiss_prop))])

            print("tiss prop", result_list)

            cfg["Domain"]["Media"] = result_list
            cfg["Domain"]["MediaFormat"] = "integer"

            # cfg['seed'] = int(np.floor(np.random.rand() * 1e+7))
            cfg["Session"]["RNGSeed"] = int(np.floor(np.random.rand() * 1e+7))
            cfg["Session"]["Photons"] = num_phot  # cfg['nphoton'] = num_phot
            cfg["Session"]["DoNormalize"] = 1
            cfg["Session"]["DoSaveVolume"] = 1
            cfg["Session"]["OutputFormat"] = 'nii'  # mc2
            cfg["Session"]["DoPartialPath"] = 1
            cfg["Session"]["DoDCS"] = 1

            cfg["Session"]["OutputType"] = 'F'  # fluence
            # cfg['issaveexit'] = 1

            cfg["Shapes"] = []
            del cfg["Optode"]["Source"]["Param1"]
            del cfg["Optode"]["Source"]["Param2"]
            del cfg["Shapes"]

            # output fluence file (.mc2), detected photon file (.mch)

            data = mcxm.run(
                cfg, "-d 1", mcxbin=r'C:\Program Files\MCXStudio\MCXSuite\mcx\bin\mcx.exe')
            # "-F jnii"

            # newdata=jd.load(cfg["Session"]["ID"]+'.jnii')
            # print('newdata', newdata)
            print('data', data)

            # flue = newdata['NIFTIData']
            # print(flue.shape)
            # flue_stat = data[0][1]
            img = nib.load(cfg["Session"]["ID"]+'.nii')
            flue = img.get_fdata()[:]
            print("nonzero check", np.any(flue[:, :, :, 0] != 0))
            print(flue.shape)
            # print(data)
            # flue, detps = mcxlab(cfg)

            # Scale the fluence
            # scale = flue['stat']['energyabs'] / np.sum(flue['data'][fwmodel['i_head']] * mua[fwmodel['i_head']]

            # TODO flue_stat['normalizer'] must be read from somewhere
            flue = flue * cfg['Forward']['Dt'] / 1  # flue_stat['normalizer']

            for jOpt in range(0, nopt):
                foo = 0
                xx, yy, zz = probe.reg_optpos[jOpt][0], probe.reg_optpos[jOpt][1], probe.reg_optpos[jOpt][2]
                while foo == 0:
                    print('xx', xx)
                    print('yy', yy)
                    print('zz', zz)
                    xx += probe.reg_optpos[jOpt][3]
                    yy += probe.reg_optpos[jOpt][4]
                    zz += probe.reg_optpos[jOpt][5]
                    foo = flue[int(np.ceil(xx)), int(
                        np.ceil(yy)), int(np.ceil(zz))]

                flueDet[ii, jOpt, iWav] = foo

            # Project to the brain surface and scalp surface

            # flueMesh[:, ii, iWav] = np.sum(np.reshape(flue[mapMesh2Vox], (nNode, maxNVoxPerNode)), axis=1)
            # print("mapmesh",np.shape(mapMesh2Vox))
            # print(flue.data[mapMesh2Vox.astype(int)])
            # flueMesh[:, ii, iWav] = np.sum(np.reshape(flue.data[mapMesh2Vox.astype(int)], (nNode, maxNVoxPerNode)), axis=1)

            mapMesh2Vox_flat = fwmodel.get_projVolToMesh_brain().flatten().astype(int)
            reshaped_data = np.reshape(
                flue.flatten()[mapMesh2Vox_flat], (nNode, -1))
            flueMesh[:, ii, iWav] = np.sum(reshaped_data, axis=1)

            mapMesh2Vox_scalp_flat = fwmodel.get_projVolToMesh_scalp().flatten().astype(int)
            reshaped_data_scalp = np.reshape(
                flue.flatten()[mapMesh2Vox_scalp_flat], (nNode_s, -1))
            flueMesh_scalp[:, ii, iWav] = np.sum(reshaped_data_scalp, axis=1)

    lst = np.where(probe.meas_list[:, 3] == 1)[0]
    nMeas = len(lst)
    # Construct Adot matrices
    Adot = np.zeros((nMeas, nNode, num_wavelengths), dtype=np.float32)
    Adot_scalp = np.zeros((nMeas, nNode_s, num_wavelengths), dtype=np.float32)

    iM2 = 0
    for iWav in range(0, num_wavelengths):
        for iM in range(0, nMeas):
            iM2 += 1

            # Load data for BRAIN
            iS = int(probe.meas_list[iM, 0])
            As = flueMesh[:, iS-1, iWav]

            iD = int(probe.meas_list[iM, 1])
            Ad = flueMesh[:, probe.nsrc + iD-1, iWav]

            normfactor = (flueDet[iS-1, probe.nsrc + iD-1, iWav] +
                          flueDet[probe.nsrc + iD-1, iS-1, iWav]) / 2

            if normfactor != 0:
                Adot[iM, :, iWav] = (As * Ad) / normfactor
            else:
                print(f'No photons detected between Src {iS} and Det {iD}')
                Adot[iM, :, iWav] = np.zeros(As.shape)

            # Load data for SCALP
            iS = int(probe.meas_list[iM, 0])
            As = flueMesh_scalp[:, iS-1, iWav]

            iD = int(probe.meas_list[iM, 1])
            Ad = flueMesh_scalp[:, probe.nsrc + iD-1, iWav]

            normfactor = (flueDet[iS-1, probe.nsrc + iD-1, iWav] +
                          flueDet[probe.nsrc + iD-1, iS-1, iWav]) / 2

            if normfactor != 0:
                Adot_scalp[iM, :, iWav] = (As * Ad) / normfactor
            else:
                print(f'No photons detected between Src {iS} and Det {iD}')
                Adot_scalp[iM, :, iWav] = np.zeros(As.shape)

    fwmodel.Adot = Adot
    fwmodel.Adot_scalp = Adot_scalp

    np.save('Adot.npy', Adot)
    np.save('Adot_scalp.npy', Adot_scalp)
