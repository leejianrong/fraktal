class_name Tile
extends Node3D
## One exported tile: relief ground, buildings, trees and poles, built from the files
## `fraktal scene` writes. Tile coordinates are x east, y north, z up, in metres from the
## south-west corner; the engine's are (x, z, -y), a rotation, so polygon winding carries over.
## Godot draws clockwise faces as front faces, so triangles are emitted reversed.

const FOLIAGE := [
	Color(0.15, 0.31, 0.13), Color(0.19, 0.36, 0.14), Color(0.12, 0.26, 0.12),
	Color(0.22, 0.38, 0.12), Color(0.17, 0.30, 0.16), Color(0.26, 0.40, 0.15),
]
const BARK := Color(0.24, 0.17, 0.11)
const FROND := Color(0.2, 0.4, 0.14)
const POLE := Color(0.33, 0.33, 0.35)
const NO_ROOF := -1.0e9

const GROUND_SHADER := """
shader_type spatial;
uniform sampler2D albedo_tex : filter_linear_mipmap, repeat_disable;
vec3 s2l(vec3 c) {
	return mix(pow((c + vec3(0.055)) / 1.055, vec3(2.4)), c / 12.92, step(c, vec3(0.04045)));
}
void fragment() {
	ALBEDO = s2l(texture(albedo_tex, UV).rgb);
	ROUGHNESS = 0.95;
}
"""

var scene := {}
var size_m := 500.0
var cell_m := 1.0
var cells := 1  # ground mesh cells per side
var nodes := PackedFloat32Array()  # (cells + 1)^2 heights, row j is y = j * cell
var stats := {}
var _footprints: Array = []  # {"box": Rect2, "ring": PackedVector2Array, "roof": float}
var _textures := {}


static func to_engine(p: Vector3) -> Vector3:
	return Vector3(p.x, p.z, -p.y)


func load_tile(scene_dir: String, assets_dir: String) -> void:
	scene = JSON.parse_string(FileAccess.get_file_as_string(scene_dir.path_join("scene.json")))
	assert(scene is Dictionary and scene.get("schema", 0) == 1, "unsupported scene.json")
	size_m = scene["size_m"]
	var t := Time.get_ticks_msec()
	_load_terrain(scene_dir)
	_lap("terrain", t)
	t = Time.get_ticks_msec()
	_build_ground(scene_dir)
	_lap("ground_mesh", t)
	t = Time.get_ticks_msec()
	_build_buildings(scene_dir, assets_dir)
	_lap("buildings", t)
	t = Time.get_ticks_msec()
	_build_trees()
	_build_poles()
	_lap("trees_poles", t)


func _lap(what: String, since: int) -> void:
	stats["ms_" + what] = Time.get_ticks_msec() - since


## Ground height, bilinear, the same rule as `blender_export.terrain_height`.
func height_at(x: float, y: float) -> float:
	var fx := clampf(x / cell_m, 0.0, cells - 1e-9)
	var fy := clampf(y / cell_m, 0.0, cells - 1e-9)
	var i := int(fx)
	var j := int(fy)
	var wx := fx - i
	var wy := fy - j
	var m := cells + 1
	return (
		nodes[j * m + i] * (1 - wx) * (1 - wy) + nodes[j * m + i + 1] * wx * (1 - wy)
		+ nodes[(j + 1) * m + i] * (1 - wx) * wy + nodes[(j + 1) * m + i + 1] * wx * wy
	)


## Top of the building over (x, y), or NO_ROOF.
func roof_at(x: float, y: float) -> float:
	var p := Vector2(x, y)
	for f in _footprints:
		if f["box"].has_point(p) and Geometry2D.is_point_in_polygon(p, f["ring"]):
			return f["roof"]
	return NO_ROOF


func _load_terrain(dir: String) -> void:
	if scene.has("terrain"):
		var t: Dictionary = scene["terrain"]
		var a := Npy.read(dir.path_join(t["file"]))
		nodes = a["data"]
		cells = a["shape"][0] - 1
		cell_m = t["cell_m"]
	else:  # a flat tile: one quad
		cells = 1
		cell_m = size_m
		nodes = PackedFloat32Array([0, 0, 0, 0])


func _build_ground(dir: String) -> void:
	var m := cells + 1
	var verts := PackedVector3Array()
	var norms := PackedVector3Array()
	var uvs := PackedVector2Array()
	verts.resize(m * m)
	norms.resize(m * m)
	uvs.resize(m * m)
	for j in m:
		for i in m:
			var k := j * m + i
			verts[k] = Vector3(i * cell_m, nodes[k], -j * cell_m)
			var i0 := maxi(i - 1, 0)
			var i1 := mini(i + 1, cells)
			var j0 := maxi(j - 1, 0)
			var j1 := mini(j + 1, cells)
			var gx := (nodes[j * m + i1] - nodes[j * m + i0]) / ((i1 - i0) * cell_m)
			var gy := (nodes[j1 * m + i] - nodes[j0 * m + i]) / ((j1 - j0) * cell_m)
			norms[k] = Vector3(-gx, 1.0, gy).normalized()  # (-gx, -gy, 1) in tile axes
			uvs[k] = Vector2(i * cell_m / size_m, 1.0 - j * cell_m / size_m)
	var idx := PackedInt32Array()
	idx.resize(cells * cells * 6)
	var w := 0
	for j in cells:
		for i in cells:
			var a := j * m + i
			idx[w] = a
			idx[w + 1] = a + m + 1
			idx[w + 2] = a + 1
			idx[w + 3] = a
			idx[w + 4] = a + m
			idx[w + 5] = a + m + 1
			w += 6
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = verts
	arrays[Mesh.ARRAY_NORMAL] = norms
	arrays[Mesh.ARRAY_TEX_UV] = uvs
	arrays[Mesh.ARRAY_INDEX] = idx
	var mesh := ArrayMesh.new()
	mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	var g := Npy.read(dir.path_join("ground.npy"))  # H x W x 3 float32, north up
	var img := Image.create_from_data(
		g["shape"][1], g["shape"][0], false, Image.FORMAT_RGBF, (g["data"] as PackedFloat32Array).to_byte_array()
	)
	img.generate_mipmaps()
	var shader := Shader.new()
	shader.code = GROUND_SHADER
	var mat := ShaderMaterial.new()
	mat.shader = shader
	mat.set_shader_parameter("albedo_tex", ImageTexture.create_from_image(img))
	mesh.surface_set_material(0, mat)
	var node := MeshInstance3D.new()
	node.name = "Ground"
	node.mesh = mesh
	add_child(node)
	stats["ground_vertices"] = m * m
	stats["ground_triangles"] = cells * cells * 2


func _build_buildings(dir: String, assets_dir: String) -> void:
	var co: PackedFloat32Array = Npy.read(dir.path_join("mesh_vertices.npy"))["data"]
	var sizes: PackedInt32Array = Npy.read(dir.path_join("mesh_face_sizes.npy"))["data"]
	var face_verts: PackedInt32Array = Npy.read(dir.path_join("mesh_face_verts.npy"))["data"]
	var face_mat: PackedInt32Array = Npy.read(dir.path_join("mesh_face_material.npy"))["data"]
	var owner: PackedInt32Array = Npy.read(dir.path_join("mesh_face_building.npy"))["data"]
	var names: Array = scene["materials"]
	var count: int = scene["buildings"].size()
	# Each building stands on the lowest ground under its footprint, as in the Blender scene.
	var base := PackedFloat32Array()
	base.resize(count)
	base.fill(1.0e9)
	var volume := PackedFloat64Array()  # signed, to find buildings whose faces wind inwards
	volume.resize(count)
	var at := 0
	for f in sizes.size():
		var b := owner[f]
		for q in sizes[f]:
			var v := face_verts[at + q]
			base[b] = minf(base[b], height_at(co[v * 3], co[v * 3 + 1]))
		var p0 := _vert(co, face_verts[at])
		for q in range(1, sizes[f] - 1):
			volume[b] += p0.dot(_vert(co, face_verts[at + q]).cross(_vert(co, face_verts[at + q + 1]))) / 6.0
		at += sizes[f]
	var flipped := 0
	for v in volume:
		if v < 0.0:
			flipped += 1
	var groups := {}  # material index -> {"v", "n", "i"}
	at = 0
	for f in sizes.size():
		var b := owner[f]
		var inward := volume[b] < 0.0
		var k := sizes[f]
		var pts: Array[Vector3] = []
		var normal := Vector3.ZERO
		for q in k:
			var v := face_verts[at + q]
			var p := _vert(co, v)
			p.z += base[b]
			pts.append(p)
		for q in k:
			normal += pts[q].cross(pts[(q + 1) % k])
		normal = normal.normalized()
		if inward:
			normal = -normal
		var en := to_engine(normal)
		var g: Dictionary = groups.get(face_mat[f], {})
		if g.is_empty():
			g = {"v": PackedVector3Array(), "n": PackedVector3Array(), "i": PackedInt32Array()}
			groups[face_mat[f]] = g
		var first: int = g["v"].size()
		for p in pts:
			g["v"].append(to_engine(p))
			g["n"].append(en)
		for q in range(1, k - 1):
			if inward:
				g["i"].append_array([first, first + q, first + q + 1])
			else:
				g["i"].append_array([first, first + q + 1, first + q])
		at += k
	var mesh := ArrayMesh.new()
	var triangles := 0
	var keys := groups.keys()
	keys.sort()
	for key in keys:
		var g: Dictionary = groups[key]
		var arrays := []
		arrays.resize(Mesh.ARRAY_MAX)
		arrays[Mesh.ARRAY_VERTEX] = g["v"]
		arrays[Mesh.ARRAY_NORMAL] = g["n"]
		arrays[Mesh.ARRAY_INDEX] = g["i"]
		mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
		mesh.surface_set_material(mesh.get_surface_count() - 1, _material(names[key], assets_dir))
		triangles += g["i"].size() / 3
	var node := MeshInstance3D.new()
	node.name = "Buildings"
	node.mesh = mesh
	add_child(node)
	# Footprints for collision: the exported rings, with the roof at base + height.
	var roofs := PackedFloat32Array()
	roofs.resize(count)
	for b in count:
		var d: Dictionary = scene["buildings"][b]
		var ring := PackedVector2Array()
		var low := 1.0e9
		for p in d["ring"]:
			ring.append(Vector2(p[0], p[1]))
			low = minf(low, height_at(p[0], p[1]))
		var floor_z := base[b] if base[b] < 1.0e8 else low
		_footprints.append({"box": _bounds(ring), "ring": ring, "roof": floor_z + float(d["height_m"])})
	stats["buildings"] = count
	stats["building_faces"] = sizes.size()
	stats["building_triangles"] = triangles
	stats["buildings_flipped"] = flipped
	stats["materials"] = names.size()


static func _vert(co: PackedFloat32Array, v: int) -> Vector3:
	return Vector3(co[v * 3], co[v * 3 + 1], co[v * 3 + 2])


static func _bounds(ring: PackedVector2Array) -> Rect2:
	var r := Rect2(ring[0], Vector2.ZERO)
	for p in ring:
		r = r.expand(p)
	return r.grow(0.01)


## The asset manifest's material (`assets/materials/NAME.json`): a colour, or a texture mapped in
## world metres and tinted. Blender reads the stored colours as linear values; Godot decodes an
## albedo colour from sRGB, so colours go through linear_to_srgb to land on the same linear value.
## (A texture is still decoded as sRGB, which Blender's non-colour image is not; see the plan doc.)
func _material(name: String, assets_dir: String) -> StandardMaterial3D:
	var spec: Dictionary = JSON.parse_string(FileAccess.get_file_as_string(assets_dir.path_join("materials/%s.json" % name)))
	var mat := StandardMaterial3D.new()
	mat.roughness = spec.get("roughness", 0.8)
	if spec.has("image"):
		var path: String = assets_dir.path_join(spec["image"])
		if not _textures.has(path):
			var img := Image.load_from_file(path)
			img.generate_mipmaps()
			_textures[path] = ImageTexture.create_from_image(img)
		mat.albedo_texture = _textures[path]
		mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
		mat.uv1_triplanar = true
		mat.uv1_world_triplanar = true
		mat.uv1_triplanar_sharpness = 4.0
		mat.uv1_scale = Vector3.ONE / float(spec["scale_m"])
		var tint: Array = spec["tint"]
		mat.albedo_color = Color(tint[0], tint[1], tint[2]).linear_to_srgb()
	else:
		var c: Array = spec["color"]
		mat.albedo_color = Color(c[0], c[1], c[2]).linear_to_srgb()
	return mat


func _multimesh(name: String, mesh: Mesh, transforms: Array[Transform3D], colours: PackedColorArray) -> void:
	if transforms.is_empty():
		return
	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.mesh = mesh
	mm.instance_count = transforms.size()
	for i in transforms.size():
		mm.set_instance_transform(i, transforms[i])
		mm.set_instance_color(i, colours[i])
	var node := MultiMeshInstance3D.new()
	node.name = name
	node.multimesh = mm
	var mat := StandardMaterial3D.new()
	mat.vertex_color_use_as_albedo = true
	mat.roughness = 0.8
	node.material_override = mat
	add_child(node)


## Stand-ins for the Blender trees: a trunk and one rounded crown each (no lumpy crowns, and a
## palm's fronds are a flat disc). Same places, sizes and species as `scene.json`.
func _build_trees() -> void:
	var trunk := CylinderMesh.new()
	trunk.top_radius = 1.0
	trunk.bottom_radius = 1.0
	trunk.height = 1.0
	trunk.radial_segments = 6
	trunk.rings = 1
	var crown := SphereMesh.new()
	crown.radius = 1.0
	crown.height = 2.0
	crown.radial_segments = 8
	crown.rings = 4
	var trunks: Array[Transform3D] = []
	var trunk_colours := PackedColorArray()
	var crowns: Array[Transform3D] = []
	var crown_colours := PackedColorArray()
	var k := 0
	for tree in scene["trees"]:
		var x: float = tree[0]
		var y: float = tree[1]
		var radius: float = tree[2]
		var height: float = tree[3]
		var species: String = tree[4] if tree.size() > 4 else "broadleaf"
		var ground := height_at(x, y)
		var trunk_r := 0.0
		var trunk_top := 0.0
		var centre := 0.0
		var rx := radius
		var rz := 0.0
		var colour: Color = FOLIAGE[(k * 5 + 1) % FOLIAGE.size()]
		if species == "palm":
			trunk_top = height * 0.9
			trunk_r = maxf(0.14, radius * 0.06)
			centre = trunk_top
			rx = maxf(radius, 1.8)
			rz = 0.3
			colour = FROND
		elif species == "shrub":
			rz = maxf(0.4, minf(height, radius * 1.4) * 0.5)
			centre = rz * 0.8
			rx = radius * 1.15
		else:
			var half := maxf(radius * 0.9, height * 0.3)
			centre = maxf(half, height - half)
			rz = half
			if centre - half > 0.4:
				trunk_top = centre - half * 0.5
				trunk_r = maxf(0.12, radius * 0.07)
		if trunk_top > 0.0:
			trunks.append(Transform3D(Basis.from_scale(Vector3(trunk_r, trunk_top, trunk_r)), Vector3(x, ground + trunk_top / 2.0, -y)))
			trunk_colours.append(BARK)
		crowns.append(Transform3D(Basis.from_scale(Vector3(rx, rz, rx)), Vector3(x, ground + centre, -y)))
		crown_colours.append(colour)
		k += 1
	_multimesh("TreeTrunks", trunk, trunks, trunk_colours)
	_multimesh("TreeCrowns", crown, crowns, crown_colours)
	stats["trees"] = crowns.size()


func _build_poles() -> void:
	var box := BoxMesh.new()
	box.size = Vector3.ONE
	var parts: Array[Transform3D] = []
	var colours := PackedColorArray()
	for p in scene["poles"]:
		var ground := height_at(p[0], p[1])
		var h: float = p[2]
		parts.append(Transform3D(Basis.from_scale(Vector3(0.2, h, 0.2)), Vector3(p[0], ground + h / 2.0, -p[1])))
		parts.append(Transform3D(Basis.from_scale(Vector3(0.9, 0.18, 0.24)), Vector3(p[0], ground + h + 0.09, -p[1])))
		colours.append(POLE)
		colours.append(POLE)
	_multimesh("Poles", box, parts, colours)
	stats["poles"] = scene["poles"].size()
