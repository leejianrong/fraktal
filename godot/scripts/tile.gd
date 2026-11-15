class_name Tile
extends Node3D
## One exported tile: relief ground, buildings, trees and poles, road markings and wear, props,
## compound walls, overhead cables and parked cars, built from the files
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
const MAX_SURFACES := 200
const CAR_BODY := [  # the Blender palette for parked cars, 0 to 255
	Color8(235, 235, 235), Color8(40, 40, 45), Color8(150, 30, 30), Color8(170, 175, 180), Color8(30, 60, 120),
]

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
var _grid := {}  # Vector2i cell -> {"segs": [[x0, y0, x1, y1, height]], "posts": [[x, y, radius]]}
const GRID_M := 8.0
## Road and pavement wear is fine detail for the Blender render. Drawn here with the other marking
## colours it shows as long pale bars across the road, so the viewer leaves it out.
const WEAR_MATERIALS := ["road-patch", "road-patch-pale", "road-stain", "road-polish", "road-crack", "manhole",
	"kerb-grime", "kerb-scuff", "pavement-stain", "pavement-crack", "pavement-hole", "weed"]


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
	t = Time.get_ticks_msec()
	_build_packed(scene_dir, assets_dir, "markings", "Markings", scene["markings"]["materials"], {}, false, WEAR_MATERIALS)
	_lap("markings", t)
	t = Time.get_ticks_msec()
	_build_packed(scene_dir, assets_dir, "props", "Props", scene["prop_materials"])
	_build_packed(scene_dir, assets_dir, "walls", "Walls", scene["wall_materials"], scene.get("wall_textures", {}))
	_lap("props_walls", t)
	t = Time.get_ticks_msec()
	_build_packed(scene_dir, assets_dir, "cables", "Cables", scene["cables"]["materials"], {}, true)
	_build_parked_cars()
	_lap("cables_cars", t)
	t = Time.get_ticks_msec()
	_build_obstacles()
	_lap("obstacles", t)


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
## world metres and tinted. Blender renders with its raw view transform, so a stored colour is the
## value shown on screen; Godot decodes an albedo colour from sRGB and tone maps, so the stored
## colour is used as an sRGB colour, as the ground shader and the textures already do.
func _material(name: String, assets_dir: String, tint_override: Array = []) -> StandardMaterial3D:
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
		var tint: Array = tint_override if not tint_override.is_empty() else spec["tint"]
		mat.albedo_color = Color(tint[0], tint[1], tint[2])
	else:
		var c: Array = spec["color"]
		mat.albedo_color = Color(c[0], c[1], c[2])
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


## One mesh from the packed arrays Blender builds from (`<prefix>_vertices.npy`, `_faces.npy` for
## triangles and `_face_material.npy`), with a surface per material. `materials` is the scene's list
## of [name, colour, roughness, metallic]; `textured` maps a name to [base material, tint] for the
## plastered walls. Like Blender, every vertex is lifted onto the ground under it. Faces are flat
## shaded (the parts are boxes and ribbons), emitted reversed for Godot's clockwise front faces.
func _build_packed(dir: String, assets_dir: String, prefix: String, node_name: String, materials: Array, textured: Dictionary = {}, double_sided := false, skip: Array = []) -> void:
	var co: PackedFloat32Array = Npy.read(dir.path_join(prefix + "_vertices.npy"))["data"]
	var tri: PackedInt32Array = Npy.read(dir.path_join(prefix + "_faces.npy"))["data"]
	var face_mat: PackedInt32Array = Npy.read(dir.path_join(prefix + "_face_material.npy"))["data"]
	if tri.is_empty():
		return
	var pts := PackedVector3Array()
	pts.resize(co.size() / 3)
	for v in pts.size():
		var x := co[v * 3]
		var y := co[v * 3 + 1]
		pts[v] = Vector3(x, co[v * 3 + 2] + height_at(x, y), -y)
	var groups := {}  # material index -> {"v", "n"}
	for f in tri.size() / 3:
		if skip.has(materials[face_mat[f]][0]):
			continue
		var a := pts[tri[f * 3]]
		var b := pts[tri[f * 3 + 1]]
		var c := pts[tri[f * 3 + 2]]
		var n := (b - a).cross(c - a).normalized()
		var g: Dictionary = groups.get(face_mat[f], {})
		if g.is_empty():
			g = {"v": PackedVector3Array(), "n": PackedVector3Array()}
			groups[face_mat[f]] = g
		g["v"].append_array(PackedVector3Array([a, c, b]))
		g["n"].append_array(PackedVector3Array([n, n, n]))
	var keys := groups.keys()
	keys.sort()
	var chunks := 0
	for start in range(0, keys.size(), MAX_SURFACES):  # a mesh holds at most 256 surfaces
		var mesh := ArrayMesh.new()
		for key in keys.slice(start, start + MAX_SURFACES):
			var g: Dictionary = groups[key]
			var arrays := []
			arrays.resize(Mesh.ARRAY_MAX)
			arrays[Mesh.ARRAY_VERTEX] = g["v"]
			arrays[Mesh.ARRAY_NORMAL] = g["n"]
			mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
			var m: Array = materials[key]
			var mat: StandardMaterial3D
			if textured.has(m[0]):
				mat = _material(textured[m[0]][0], assets_dir, textured[m[0]][1])
			else:
				mat = StandardMaterial3D.new()
				mat.albedo_color = Color(m[1][0], m[1][1], m[1][2])  # display values, like the ground
				mat.roughness = m[2]
				mat.metallic = m[3]
			if double_sided:
				mat.cull_mode = BaseMaterial3D.CULL_DISABLED
			mesh.surface_set_material(mesh.get_surface_count() - 1, mat)
		var node := MeshInstance3D.new()
		node.name = node_name if chunks == 0 else "%s%d" % [node_name, chunks + 1]
		node.mesh = mesh
		add_child(node)
		chunks += 1
	stats[prefix + "_triangles"] = tri.size() / 3


## The cars parked along the road (`scene["vehicles"]`: x, y, heading, colour): a body and a glass
## cabin, as boxes, like the Blender scene's before it swaps them for moving traffic.
func _build_parked_cars() -> void:
	var cars: Array = scene.get("vehicles", [])
	var box := BoxMesh.new()
	box.size = Vector3.ONE
	var bodies: Array[Transform3D] = []
	var body_colours := PackedColorArray()
	var cabins: Array[Transform3D] = []
	var cabin_colours := PackedColorArray()
	for car in cars:
		var x: float = car[0]
		var y: float = car[1]
		var heading: float = car[2]
		var basis := Basis(Vector3.UP, heading)
		var ground := height_at(x, y)
		var centre := Vector3(x, ground, -y)
		var back: Vector3 = basis * Vector3(-0.2, 0.0, 0.0)
		bodies.append(Transform3D(basis * Basis.from_scale(Vector3(4.5, 0.75, 1.9)), centre + Vector3(0, 0.625, 0)))
		body_colours.append(CAR_BODY[int(car[3]) % CAR_BODY.size()])
		cabins.append(Transform3D(basis * Basis.from_scale(Vector3(2.4, 0.5, 1.7)), centre + back + Vector3(0, 1.25, 0)))
		cabin_colours.append(Color(0.05, 0.07, 0.1))
	for car in cars:  # a parked car stops a person or a car: three posts along its length
		for along in [-1.4, 0.0, 1.4]:
			_cell_entry(_cell(car[0], car[1]))["posts"].append(
				[car[0] + cos(car[2]) * along, car[1] + sin(car[2]) * along, 0.95])
	_multimesh("ParkedCars", box, bodies, body_colours)
	_multimesh("ParkedCabins", box, cabins, cabin_colours)
	stats["parked_cars"] = cars.size()


func _cell(x: float, y: float) -> Vector2i:
	return Vector2i(int(floorf(x / GRID_M)), int(floorf(y / GRID_M)))


func _cell_entry(c: Vector2i) -> Dictionary:
	if not _grid.has(c):
		_grid[c] = {"segs": [], "posts": []}
	return _grid[c]


## What a person or low drone cannot walk through besides buildings: compound walls (segments from
## `scene["walls"]`, each with a height), tree trunks and poles (as posts). A grid of 8 m cells keeps
## the per-step test to a few entries.
func _build_obstacles() -> void:
	var segs := 0
	for wall in scene.get("walls", []):
		var h: float = wall["height_m"]
		for sg in wall["segments"]:
			var seg := [sg[0], sg[1], sg[2], sg[3], h]
			var steps := int(Vector2(sg[2] - sg[0], sg[3] - sg[1]).length() / (GRID_M * 0.5)) + 1
			var seen := {}
			for i in steps + 1:
				var t := float(i) / steps
				var c := _cell(lerpf(sg[0], sg[2], t), lerpf(sg[1], sg[3], t))
				for dx in [-1, 0, 1]:
					for dy in [-1, 0, 1]:
						var key := c + Vector2i(dx, dy)
						if not seen.has(key):
							seen[key] = true
							_cell_entry(key)["segs"].append(seg)
			segs += 1
	var posts := 0
	for tree in scene["trees"]:
		var species: String = tree[4] if tree.size() > 4 else "broadleaf"
		if species == "shrub":
			continue
		_cell_entry(_cell(tree[0], tree[1]))["posts"].append([tree[0], tree[1], maxf(0.14, tree[2] * 0.07)])
		posts += 1
	for pole in scene["poles"]:
		_cell_entry(_cell(pole[0], pole[1]))["posts"].append([pole[0], pole[1], 0.15])
		posts += 1
	stats["wall_segments"] = segs
	stats["posts"] = posts


## Whether something of this radius at (x, y) is in a compound wall (while it is taller than 0.5 m),
## a trunk, a pole or a parked car.
func static_hit(x: float, y: float, radius: float) -> bool:
	var entry: Dictionary = _grid.get(_cell(x, y), {})
	var p := Vector2(x, y)
	for sg in entry.get("segs", []):
		if sg[4] > 0.5 and Geometry2D.get_closest_point_to_segment(p, Vector2(sg[0], sg[1]), Vector2(sg[2], sg[3])).distance_to(p) < radius + 0.1:
			return true
	var c := _cell(x, y)
	for dx in [-1, 0, 1]:  # a post is kept in its own cell only, and may reach into the next
		for dy in [-1, 0, 1]:
			for post in _grid.get(c + Vector2i(dx, dy), {}).get("posts", []):
				if p.distance_to(Vector2(post[0], post[1])) < radius + post[2]:
					return true
	return false


## Whether a moving car or a person is within `radius` of (x, y). Set by main.gd once the agents load.
var crowd: Agents = null


func crowd_hit(x: float, y: float, radius: float) -> bool:
	return crowd != null and crowd.hit(x, y, radius)


## Whether a person of this radius standing at (x, y) is in something: a building, a wall, a trunk,
## a pole, a parked car, or another person or car.
func person_blocked(x: float, y: float, radius: float) -> bool:
	if x < 0.0 or y < 0.0 or x > size_m or y > size_m:
		return true
	for d in [Vector2(0, 0), Vector2(radius, 0), Vector2(-radius, 0), Vector2(0, radius), Vector2(0, -radius)]:
		if roof_at(x + d.x, y + d.y) != NO_ROOF:
			return true
	return static_hit(x, y, radius) or crowd_hit(x, y, radius)
