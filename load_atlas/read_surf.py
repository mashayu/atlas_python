import numpy as np
"""
    def read_surf(fname):
        # Open the file as a big-endian binary file
        with open(fname, "rb") as fid:
            # Read the magic number
            magic = int.from_bytes(fid.read(3), byteorder="big")

            if magic == 16777215:  # QUAD_FILE_MAGIC_NUMBER
                vnum = int.from_bytes(fid.read(3), byteorder="big")
                fnum = int.from_bytes(fid.read(3), byteorder="big")
                vertex_coords = np.fromfile(fid, dtype=np.int16, count=vnum * 3) / 100

                # Read faces if requested
                if nargout > 1:
                    faces = np.zeros((fnum, 4), dtype=int)
                    for i in range(fnum):
                        for n in range(4):
                            faces[i, n] = int.from_bytes(fid.read(3), byteorder="big")

            elif magic == 16777214:  # TRIANGLE_FILE_MAGIC_NUMBER
                fid.readline()
                fid.readline()
                vnum = int.from_bytes(fid.read(4), byteorder="big")
                fnum = int.from_bytes(fid.read(4), byteorder="big")
                vertex_coords = np.fromfile(fid, dtype=np.float32, count=vnum * 3)
                faces = np.fromfile(fid, dtype=int, count=fnum * 3).reshape(-1, 3)

        # Increment face indices by 1 to make them 1-based
        # faces = faces + 1
        vertex_coords = vertex_coords.reshape(-1, 3)

        return vertex_coords, faces


    # Example usage:
    # vertex_coords, faces = read_surf('your_surface_file.surf')


    def read_surf2(fname):
        with open(fname, "rb") as fid:
            # Read magic number
            magic = int.from_bytes(fid.read(3), byteorder="big")

            if magic != 16777214:
                raise ValueError("Not a valid FreeSurfer triangle file.")

            # Skip the header string
            while True:
                line = fid.readline()
                if not line.strip():  # Empty line indicates end of header
                    break

            # Read number of vertices and faces
            nv = int.from_bytes(fid.read(4), byteorder="big")
            nf = int.from_bytes(fid.read(4), byteorder="big")

            # Read vertex coordinates
            vertex_coords = np.fromfile(fid, dtype=np.float32, count=nv * 3).reshape(nv, 3)

            # Read faces
            faces = np.fromfile(fid, dtype=np.int, count=nf * 3)
            print(faces.shape)
            print(faces)
            faces = faces.reshape(nf, 3)

            # Adjust for 0-based indexing in Python (since MATLAB uses 1-based)
            faces = faces + 1

        return vertex_coords, faces
"""
