"""
BigWorld .vt Collision Mesh Importer for Blender
-------------------------------------------------
Supports Blender 2.80 and later.
Author : Icefire00

Installation:
  Edit > Preferences > Add-ons > Install...
  Select this file, then enable "Import-Export: BigWorld VT Collision Mesh".

Usage:
  File > Import > BigWorld VT Collision (.vt)

Format notes (reverse-engineered from obj2vt.py in github.com/atacms/wot-vt-converter/):
  Offset  Size  Type      Description
  ------  ----  --------  -----------
  0       4     uint32    Magic  = 0xB00BB00B  (bytes: 0B B0 0B B0)
  4       4     uint32    Version = 2
  8       24    6×float   Bounding box: minx miny minz maxx maxy maxz
  32      4     uint32    Vertex count (N)
  36      N×12  N×3float  Vertices (x, y, z)  — z was negated when written
  *       4     uint32    Index count (M)
  *       1     uint8     Index type: 0x01 = uint16, 0x02 = uint32
  *       M×?   indices   Triangle indices (uint16 or uint32)
  *       8     bytes     Footer flag: 01 00 00 00 00 00 00 00
  *       4     uint32    Vertex count again (redundant)
"""

bl_info = {
    "name":        "BigWorld VT Collision Mesh",
    "author":      "Generated Importer",
    "version":     (1, 0, 0),
    "blender":     (5, 0, 0),
    "location":    "File > Import > BigWorld VT Collision (.vt)",
    "description": "Import WoT BigWorld engine's .vt collision mesh files",
    "category":    "Import-Export",
}

import os
import struct

import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper

# ---------------------------------------------------------------------------
# Binary reader helper
# ---------------------------------------------------------------------------

class BinaryReader:
    """Thin wrapper around bytes for sequential reading with error reporting."""

    def __init__(self, data: bytes, filepath: str = ""):
        self._data = data
        self._pos  = 0
        self._path = filepath

    @property
    def pos(self):
        return self._pos

    def remaining(self):
        return len(self._data) - self._pos

    def read_raw(self, n: int) -> bytes:
        if self._pos + n > len(self._data):
            raise EOFError(
                f"[{self._path}] Tried to read {n} bytes at offset "
                f"0x{self._pos:08X} but only {self.remaining()} bytes remain."
            )
        chunk = self._data[self._pos : self._pos + n]
        self._pos += n
        return chunk

    def read_uint32(self) -> int:
        return struct.unpack_from("<I", self.read_raw(4))[0]

    def read_uint16(self) -> int:
        return struct.unpack_from("<H", self.read_raw(2))[0]

    def read_uint8(self) -> int:
        return struct.unpack_from("<B", self.read_raw(1))[0]

    def read_float(self) -> float:
        return struct.unpack_from("<f", self.read_raw(4))[0]

    def read_floats(self, count: int):
        fmt = f"<{count}f"
        size = 4 * count
        return struct.unpack_from(fmt, self.read_raw(size))

    def read_uint16s(self, count: int):
        fmt = f"<{count}H"
        size = 2 * count
        return struct.unpack_from(fmt, self.read_raw(size))

    def read_uint32s(self, count: int):
        fmt = f"<{count}I"
        size = 4 * count
        return struct.unpack_from(fmt, self.read_raw(size))


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

VT_MAGIC   = 0xB00BB00B   # little-endian interpretation of bytes 0B B0 0B B0
VT_VERSION = 2


def parse_vt(filepath: str):
    """
    Parse a BigWorld .vt file and return (vertices, faces, bbox).

    vertices : list of (x, y, z) float tuples  (z un-negated back to OBJ space)
    faces    : list of (i0, i1, i2) int tuples
    bbox     : (minx, miny, minz, maxx, maxy, maxz)
    """
    with open(filepath, "rb") as f:
        data = f.read()

    r = BinaryReader(data, filepath)

    # --- Header ---
    magic   = r.read_uint32()
    version = r.read_uint32()

    if magic != VT_MAGIC:
        raise ValueError(
            f"Bad magic: expected 0x{VT_MAGIC:08X}, got 0x{magic:08X}.\n"
            f"File: {filepath}"
        )
    if version != VT_VERSION:
        # Warn but continue; the format may still be compatible.
        print(f"[VT Importer] Warning: version {version} (expected {VT_VERSION}).")

    # --- Bounding box (informational only) ---
    minx, miny, minz, maxx, maxy, maxz = r.read_floats(6)
    bbox = (minx, miny, minz, maxx, maxy, maxz)

    # --- Vertices ---
    v_count  = r.read_uint32()
    raw_verts = r.read_floats(v_count * 3)

    # Full BigWorld -> Blender transform XZY -> XYZ:(bw_x, -bw_z,  bw_y)

    vertices = []
    for i in range(v_count):
        bw_x = raw_verts[i * 3]
        bw_y = raw_verts[i * 3 + 1]
        bw_z = raw_verts[i * 3 + 2]
        vertices.append((bw_x, bw_z, bw_y))
    # --- Indices ---
    idx_count  = r.read_uint32()
    index_type = r.read_uint8()   # 0x01 = uint16, 0x02 = uint32

    if index_type == 0x01:
        indices = list(r.read_uint16s(idx_count))
    elif index_type == 0x02:
        indices = list(r.read_uint32s(idx_count))
    else:
        raise ValueError(
            f"Unknown index type byte 0x{index_type:02X} at offset 0x{r.pos - 1:08X}."
        )

    if idx_count % 3 != 0:
        raise ValueError(
            f"Index count {idx_count} is not divisible by 3 — not a triangle mesh?"
        )

    faces = [
        (indices[i], indices[i + 2], indices[i + 1])   # reversed winding = flipped normals
        for i in range(0, idx_count, 3)
    ]

    return vertices, faces, bbox


# ---------------------------------------------------------------------------
# Blender mesh builder
# ---------------------------------------------------------------------------

def build_blender_mesh(name: str, vertices, faces, bbox, import_bbox: bool):
    """Create a Blender mesh object from parsed VT data."""

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    mesh.validate(verbose=True)

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Optionally add an empty to visualise the stored bounding box
    if import_bbox:
        minx, miny, minz, maxx, maxy, maxz = bbox
        # Apply the same XZY → XYZ conversion used for vertices:
        #   blender_x =  bw_x,  blender_y = -bw_z,  blender_z = bw_y
        cx = (minx + maxx) / 2
        cy = -((minz + maxz) / 2)   # un-negate stored Z → Blender Y
        cz =  (miny + maxy) / 2     # stored Y → Blender Z

        empty = bpy.data.objects.new(f"{name}_bbox", None)
        empty.empty_display_type  = "CUBE"
        empty.empty_display_size  = max(maxx - minx, maxy - miny, maxz - minz) / 2
        empty.location            = (cx, cy, cz)
        bpy.context.collection.objects.link(empty)
        empty.parent = obj

    return obj


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class ImportBigWorldVT(bpy.types.Operator, ImportHelper):
    """Import a BigWorld .vt collision mesh"""

    bl_idname  = "import_mesh.bigworld_vt"
    bl_label   = "Import BigWorld VT (.vt)"
    bl_options = {"REGISTER", "UNDO"}

    # ImportHelper supplies the 'filepath' property.
    filename_ext = ".vt"
    filter_glob: StringProperty(default="*.vt", options={"HIDDEN"})

    # Allow selecting multiple files at once
    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )
    directory: StringProperty(subtype="DIR_PATH")

    import_bbox: BoolProperty(
        name="Visualise Stored BBox",
        description=(
            "Add a child Empty that shows the bounding box "
            "stored inside the .vt file"
        ),
        default=False,
    )

    def execute(self, context):
        # Support both single-file and multi-file selection
        if self.files:
            paths = [os.path.join(self.directory, f.name) for f in self.files]
        else:
            paths = [self.filepath]

        imported = 0
        errors   = []

        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            try:
                vertices, faces, bbox = parse_vt(path)
                build_blender_mesh(name, vertices, faces, bbox, self.import_bbox)
                self.report(
                    {"INFO"},
                    f"Imported '{name}': {len(vertices)} verts, "
                    f"{len(faces)} tris.",
                )
                imported += 1
            except Exception as exc:
                msg = f"Failed to import '{name}': {exc}"
                self.report({"ERROR"}, msg)
                errors.append(msg)

        if errors:
            return {"CANCELLED"} if imported == 0 else {"FINISHED"}
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Menu entry & registration
# ---------------------------------------------------------------------------

def menu_func_import(self, context):
    self.layout.operator(
        ImportBigWorldVT.bl_idname,
        text="BigWorld VT Collision (.vt)",
    )

_classes = (ImportBigWorldVT,)

def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":

    register()
