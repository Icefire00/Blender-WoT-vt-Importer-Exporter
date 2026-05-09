"""
BigWorld .vt Collision Mesh Exporter for Blender
-------------------------------------------------
Should supports Blender 2.80 and later.
Author : Icefire00

Installation:
  Edit > Preferences > Add-ons > Install...
  Select this file, then enable "Import-Export: BigWorld VT Collision Mesh Exporter".

Usage:
  Select the mesh object(s) you want to export, then:
  File > Export > BigWorld VT Collision (.vt)

Coordinate transform (inverse of the importer):
  Blender → BigWorld
    vt_x =  bl_x          (X unchanged)
    vt_y =  bl_z          (Blender Z  → BigWorld Y)
    vt_z =  bl_y          (Blender Y  → BigWorld Z)
  Winding order is reversed on export (equivalent to flipping normals),
  matching the round-trip behaviour of the importer.

Format written:(reverse-engineered from obj2vt.py in github.com/atacms/wot-vt-converter/)
  Offset  Size  Type      Description
  ------  ----  --------  -----------
  0       4     uint32    Magic  = 0xB00BB00B
  4       4     uint32    Version = 2
  8       24    6×float   Bounding box: minx miny minz maxx maxy maxz
  32      4     uint32    Vertex count (N)
  36      N×12  N×3float  Vertices (x, y, z) in BigWorld space
  *       4     uint32    Index count (M = triangles × 3)
  *       1     uint8     Index type: 0x01 = uint16 (≤65535 verts), 0x02 = uint32
  *       M×?   indices   Triangle indices
  *       8     bytes     Footer: 01 00 00 00 00 00 00 00
  *       4     uint32    Vertex count again
"""

bl_info = {
    "name":        "BigWorld VT Collision Mesh Exporter",
    "author":      "Generated Exporter",
    "version":     (1, 0, 0),
    "blender":     (5, 0, 0),
    "location":    "File > Export > BigWorld VT Collision (.vt)",
    "description": "Export selected mesh as a BigWorld engine .vt collision file",
    "category":    "Import-Export",
}

import os
import struct

import bpy
import bmesh
from bpy.props import StringProperty, BoolProperty, EnumProperty
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VT_MAGIC   = 0xB00BB00B
VT_VERSION = 2


# ---------------------------------------------------------------------------
# Core writer
# ---------------------------------------------------------------------------

def mesh_to_vt_data(obj, apply_modifiers: bool):
    """
    Triangulate the mesh of *obj* and return (vertices, indices) in BigWorld space.

    Blender → BigWorld coordinate transform (inverse of importer):
        vt_x =  bl_x
        vt_y =  bl_z   (swap Y↔Z)
        vt_z =  bl_y
    Winding is reversed: [i0, i1, i2] → [i0, i2, i1]

    Returns:
        vertices : list of (vt_x, vt_y, vt_z) floats  (world-space)
        indices  : flat list of ints  (groups of 3 = one triangle)
    """
    # Get a triangulated mesh in world space
    depsgraph = bpy.context.evaluated_depsgraph_get()
    if apply_modifiers:
        eval_obj = obj.evaluated_get(depsgraph)
    else:
        eval_obj = obj

    tmp_mesh = eval_obj.to_mesh()

    # Apply world matrix so the .vt is in world space (matching original exporter)
    world = obj.matrix_world
    tmp_mesh.transform(world)

    # Triangulate with bmesh
    bm = bmesh.new()
    bm.from_mesh(tmp_mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(tmp_mesh)
    bm.free()

    tmp_mesh.calc_loop_triangles()

    # Build vertex list in BigWorld space
    vertices = []
    for v in tmp_mesh.vertices:
        bl_x, bl_y, bl_z = v.co
        vt_x =  bl_x
        vt_y =  bl_z   # Blender Z → BigWorld Y
        vt_z =  bl_y   # Blender Y → BigWorld Z
        vertices.append((vt_x, vt_y, vt_z))

    # Build index list with reversed winding
    indices = []
    for tri in tmp_mesh.loop_triangles:
        i0, i1, i2 = tri.vertices
        indices.extend([i0, i2, i1])   # reversed winding = flipped normals

    eval_obj.to_mesh_clear()

    return vertices, indices


def compute_bbox(vertices):
    """Return (minx, miny, minz, maxx, maxy, maxz) from a list of (x,y,z) tuples."""
    if not vertices:
        return (0.0,) * 6
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return min(xs), min(ys), min(zs), max(xs), max(ys), max(zs)


def write_vt(filepath: str, vertices, indices):
    """Serialise vertices and indices to a BigWorld .vt binary file."""

    v_count  = len(vertices)
    idx_count = len(indices)
    use_32bit = v_count > 65535

    minx, miny, minz, maxx, maxy, maxz = compute_bbox(vertices)

    with open(filepath, "wb") as f:
        # Header
        f.write(struct.pack("<I", VT_MAGIC))
        f.write(struct.pack("<I", VT_VERSION))

        # Bounding box
        f.write(struct.pack("<6f", minx, miny, minz, maxx, maxy, maxz))

        # Vertices
        f.write(struct.pack("<I", v_count))
        for vt_x, vt_y, vt_z in vertices:
            f.write(struct.pack("<3f", vt_x, vt_y, vt_z))

        # Indices
        f.write(struct.pack("<I", idx_count))
        if use_32bit:
            f.write(b"\x02")
            for idx in indices:
                f.write(struct.pack("<I", idx))
        else:
            f.write(b"\x01")
            for idx in indices:
                f.write(struct.pack("<H", idx))

        # Footer
        f.write(b"\x01\x00\x00\x00\x00\x00\x00\x00")
        f.write(struct.pack("<I", v_count))

    return v_count, idx_count // 3, use_32bit


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class ExportBigWorldVT(bpy.types.Operator, ExportHelper):
    """Export selected mesh(es) as BigWorld .vt collision file(s)"""

    bl_idname  = "export_mesh.bigworld_vt"
    bl_label   = "Export BigWorld VT (.vt)"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".vt"
    filter_glob: StringProperty(default="*.vt", options={"HIDDEN"})

    apply_modifiers: BoolProperty(
        name="Apply Modifiers",
        description="Apply modifiers before exporting",
        default=True,
    )

    export_selection: EnumProperty(
        name="Export",
        description="Which objects to export",
        items=[
            ("SELECTED", "Selected Objects",
             "Export each selected mesh as a separate .vt file"),
            ("ACTIVE",   "Active Object Only",
             "Export only the active object to the chosen filename"),
        ],
        default="ACTIVE",
    )

    def execute(self, context):
        if self.export_selection == "ACTIVE":
            objects = [context.active_object] if context.active_object else []
        else:
            objects = [o for o in context.selected_objects if o.type == "MESH"]

        if not objects:
            self.report({"ERROR"}, "No mesh object selected.")
            return {"CANCELLED"}

        # Filter to mesh objects only
        mesh_objects = [o for o in objects if o.type == "MESH"]
        if not mesh_objects:
            self.report({"ERROR"}, "Selected object(s) have no mesh data.")
            return {"CANCELLED"}

        base_path = os.path.splitext(self.filepath)[0]
        exported = 0
        errors   = []

        for obj in mesh_objects:
            # If exporting multiple objects, suffix each with the object name
            if len(mesh_objects) > 1:
                out_path = f"{base_path}_{obj.name}.vt"
            else:
                out_path = self.filepath

            try:
                vertices, indices = mesh_to_vt_data(obj, self.apply_modifiers)

                if not vertices:
                    raise ValueError("Mesh has no vertices after evaluation.")
                if not indices:
                    raise ValueError("Mesh has no faces after triangulation.")

                v_count, tri_count, used_32bit = write_vt(out_path, vertices, indices)

                index_type = "uint32" if used_32bit else "uint16"
                self.report(
                    {"INFO"},
                    f"Exported '{obj.name}' → {os.path.basename(out_path)}: "
                    f"{v_count} verts, {tri_count} tris [{index_type} indices].",
                )
                exported += 1

            except Exception as exc:
                msg = f"Failed to export '{obj.name}': {exc}"
                self.report({"ERROR"}, msg)
                errors.append(msg)

        if errors:
            return {"CANCELLED"} if exported == 0 else {"FINISHED"}
        return {"FINISHED"}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "export_selection")
        layout.prop(self, "apply_modifiers")


# ---------------------------------------------------------------------------
# Menu entry & registration
# ---------------------------------------------------------------------------

def menu_func_export(self, context):
    self.layout.operator(
        ExportBigWorldVT.bl_idname,
        text="BigWorld VT Collision (.vt)",
    )


_classes = (ExportBigWorldVT,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
