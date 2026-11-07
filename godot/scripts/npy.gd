class_name Npy
## Reads the NumPy .npy files that `fraktal scene` writes: version 1 or 2, little endian float32,
## int32 or int16, C order or (for 1-D and 2-D arrays) Fortran order, which NumPy writes when an
## array came out of transposed indexing, as `ground_height.npy` does. The result is always
## row-major. Returns {"shape": PackedInt32Array, "data": a packed array}.


static func read(path: String) -> Dictionary:
	var bytes := FileAccess.get_file_as_bytes(path)
	assert(bytes.size() > 10, "cannot read %s" % path)
	assert(bytes[0] == 0x93 and bytes.slice(1, 6).get_string_from_ascii() == "NUMPY", "not an npy file: %s" % path)
	var start := 10
	var hlen := bytes.decode_u16(8)
	if bytes[6] >= 2:
		start = 12
		hlen = bytes.decode_u32(8)
	var header := bytes.slice(start, start + hlen).get_string_from_ascii()
	var fortran := "'fortran_order': True" in header
	var descr := RegEx.create_from_string("'descr': '([^']+)'").search(header).get_string(1)
	var shape := PackedInt32Array()
	for part in RegEx.create_from_string("'shape': \\(([^)]*)\\)").search(header).get_string(1).split(","):
		if part.strip_edges() != "":
			shape.append(int(part))
	var raw := bytes.slice(start + hlen)
	var data: Variant
	match descr:
		"<f4":
			data = raw.to_float32_array()
		"<i4":
			data = raw.to_int32_array()
		"<i2":
			var out := PackedInt32Array()
			out.resize(raw.size() / 2)
			for i in out.size():
				out[i] = raw.decode_s16(i * 2)
			data = out
		_:
			assert(false, "unsupported dtype %s in %s" % [descr, path])
	if fortran and shape.size() == 2:
		data = _transposed(data, shape[0], shape[1])
	else:
		assert(not fortran or shape.size() < 2, "%s: Fortran order is only handled for 2-D arrays" % path)
	return {"shape": shape, "data": data}


## Row-major copy of a column-major (rows x cols) array.
static func _transposed(src: Variant, rows: int, cols: int) -> Variant:
	var out: Variant = src.duplicate()
	for j in rows:
		for i in cols:
			out[j * cols + i] = src[i * rows + j]
	return out
