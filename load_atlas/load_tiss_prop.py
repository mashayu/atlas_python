import numpy as np
import os


def get_tiss_prop(*args):
    tiss_prop = []

    # Default tissue property values
    SCATTERING_SKIN_DEF_VAL = 0.6600
    SCATTERING_SKULL_DEF_VAL = 0.8600
    SCATTERING_DM_DEF_VAL = 0.6600
    SCATTERING_CSF_DEF_VAL = 0.0100
    SCATTERING_GM_DEF_VAL = 1.1000
    SCATTERING_WM_DEF_VAL = 1.1000
    SCATTERING_OTHER_DEF_VAL = 0.8600

    ABSORPTION_SKIN_DEF_VAL = 0.0191
    ABSORPTION_SKULL_DEF_VAL = 0.0136
    ABSORPTION_DM_DEF_VAL = 0.0191
    ABSORPTION_CSF_DEF_VAL = 0.0026
    ABSORPTION_GM_DEF_VAL = 0.0186
    ABSORPTION_WM_DEF_VAL = 0.0186
    ABSORPTION_OTHER_DEF_VAL = 0.0191

    ANISOTROPY_SKIN_DEF_VAL = 0.0010
    ANISOTROPY_SKULL_DEF_VAL = 0.0010
    ANISOTROPY_DM_DEF_VAL = 0.0010
    ANISOTROPY_CSF_DEF_VAL = 0.0010
    ANISOTROPY_GM_DEF_VAL = 0.0010
    ANISOTROPY_WM_DEF_VAL = 0.0010
    ANISOTROPY_OTHER_DEF_VAL = 0.0010

    REFRACTION_SKIN_DEF_VAL = 1.0000
    REFRACTION_SKULL_DEF_VAL = 1.0000
    REFRACTION_DM_DEF_VAL = 1.0000
    REFRACTION_CSF_DEF_VAL = 1.0000
    REFRACTION_GM_DEF_VAL = 1.0000
    REFRACTION_WM_DEF_VAL = 1.0000
    REFRACTION_OTHER_DEF_VAL = 1.0000

    # Extract args
    if len(args) >= 1:
        if os.path.exists(args[0]):
            maxiter = 20
            iter_count = 0
            filename = args[0]
            names = []
            with open(filename, "rt") as fid:
                for line in fid:
                    iter_count += 1
                    names.append(line.strip())
                    if iter_count >= maxiter:
                        break

        else:
            if not isinstance(names0, list):
                names = []
                iname = ([0] + [i for i, c in enumerate(names0) if c == ":"] +
                         [len(names0)])
                for i in range(len(iname) - 1):
                    j = iname[i] + 1
                    k = iname[i + 1]
                    names.append(names0[j:k])
            else:
                names = names0

    propnames = ["scattering", "anisotropy", "absorption", "refraction"]

    if len(args) == 2 and isinstance(args[1], str):
        propnames0 = args[1]
        # Exract and separate all property names into cells
        if ":" in propnames0:
            propnames = propnames0.split(":")
        else:
            propnames = [propnames0]

    # Parse tissue names and tissue property names and find their values
    propval = np.zeros((len(names), len(propnames)))
    for i in range(len(propnames)):
        propname = propnames[i].lower()
        for j in range(len(names)):
            name = names[j].lower()
            if propname == "anisotropy":
                if name in ["skin", "scalp"]:
                    propval[j, i] = ANISOTROPY_SKIN_DEF_VAL
                elif name in ["skull", "bone"]:
                    propval[j, i] = ANISOTROPY_SKULL_DEF_VAL
                elif name in ["dura", "dura mater", "dm"]:
                    propval[j, i] = ANISOTROPY_DM_DEF_VAL
                elif name in ["csf", "cerebral spinal fluid"]:
                    propval[j, i] = ANISOTROPY_CSF_DEF_VAL
                elif name in ["gm", "gray matter", "brain"]:
                    propval[j, i] = ANISOTROPY_GM_DEF_VAL
                elif name in ["wm", "white matter"]:
                    propval[j, i] = ANISOTROPY_WM_DEF_VAL
                else:
                    propval[j, i] = ANISOTROPY_OTHER_DEF_VAL
            elif propname == "scattering":
                if name in ["skin", "scalp"]:
                    propval[j, i] = SCATTERING_SKIN_DEF_VAL
                elif name in ["skull", "bone"]:
                    propval[j, i] = SCATTERING_SKULL_DEF_VAL
                elif name in ["dura", "dura mater", "dm"]:
                    propval[j, i] = SCATTERING_DM_DEF_VAL
                elif name in ["csf", "cerebral spinal fluid"]:
                    propval[j, i] = SCATTERING_CSF_DEF_VAL
                elif name in ["gm", "gray matter", "brain"]:
                    propval[j, i] = SCATTERING_GM_DEF_VAL
                elif name in ["wm", "white matter"]:
                    propval[j, i] = SCATTERING_WM_DEF_VAL
                else:
                    propval[j, i] = SCATTERING_OTHER_DEF_VAL
            elif propname == "absorption":
                if name in ["skin", "scalp"]:
                    propval[j, i] = ABSORPTION_SKIN_DEF_VAL
                elif name in ["skull", "bone"]:
                    propval[j, i] = ABSORPTION_SKULL_DEF_VAL
                elif name in ["dura", "dura mater", "dm"]:
                    propval[j, i] = ABSORPTION_DM_DEF_VAL
                elif name in ["csf", "cerebral spinal fluid"]:
                    propval[j, i] = ABSORPTION_CSF_DEF_VAL
                elif name in ["gm", "gray matter", "brain"]:
                    propval[j, i] = ABSORPTION_GM_DEF_VAL
                elif name in ["wm", "white matter"]:
                    propval[j, i] = ABSORPTION_WM_DEF_VAL
                else:
                    propval[j, i] = ABSORPTION_OTHER_DEF_VAL
            elif propname == "refraction":
                if name in ["skin", "scalp"]:
                    propval[j, i] = REFRACTION_SKIN_DEF_VAL
                elif name in ["skull", "bone"]:
                    propval[j, i] = REFRACTION_SKULL_DEF_VAL
                elif name in ["dura", "dura mater", "dm"]:
                    propval[j, i] = REFRACTION_DM_DEF_VAL
                elif name in ["csf", "cerebral spinal fluid"]:
                    propval[j, i] = REFRACTION_CSF_DEF_VAL
                elif name in ["gm", "gray matter", "brain"]:
                    propval[j, i] = REFRACTION_GM_DEF_VAL
                elif name in ["wm", "white matter"]:
                    propval[j, i] = REFRACTION_WM_DEF_VAL
                else:
                    propval[j, i] = REFRACTION_OTHER_DEF_VAL

    # Assign results to output struct
    for j in range(len(names)):
        tiss_struct = {
            "name": names[j],
        }
        for i in range(len(propnames)):
            tiss_struct[propnames[i]] = propval[j, i]
        tiss_prop.append(tiss_struct)

    return tiss_prop


# Example usage:
# tiss_prop = get_tiss_prop('skin:skull:csf', 'absorption')
