#[vertex]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(location = 0) out vec3 vertex_interp;
layout(location = 1) out vec3 normal_interp;
layout(location = 2) out vec2 uv_interp;
layout(location = 3) out vec3 barycentric;
layout(location = 4) flat out uvec3 vertex_indices;
layout(location = 5) flat out vec3 face_normal;
layout(location = 6) flat out uint fragment_action;
layout(location = 7) flat out vec2 v1_uv;
layout(location = 8) flat out vec2 v2_uv;
layout(location = 9) flat out vec2 v3_uv;
layout(location = 10) flat out vec3 v1_pos;
layout(location = 11) flat out vec3 v2_pos;
layout(location = 12) flat out vec3 v3_pos;

layout(push_constant, std430) uniform Params {
	vec2 atlas_size;
	vec2 uv_offset;
	vec3 to_cell_size;
	uint base_triangle;
	vec3 to_cell_offset;
	float bias;
	ivec3 grid_size;
	uint pad2;
}
params;

void main() {
	uint triangle_idx = params.base_triangle + gl_VertexIndex / 3;
	uint triangle_subidx = gl_VertexIndex % 3;

	vertex_indices = triangles.data[triangle_idx].indices;

	uint vertex_idx;
	if (triangle_subidx == 0) {
		vertex_idx = vertex_indices.x;
		barycentric = vec3(1, 0, 0);
	} else if (triangle_subidx == 1) {
		vertex_idx = vertex_indices.y;
		barycentric = vec3(0, 1, 0);
	} else {
		vertex_idx = vertex_indices.z;
		barycentric = vec3(0, 0, 1);
	}

	vertex_interp = vertices.data[vertex_idx].position;
	uv_interp = vertices.data[vertex_idx].uv;
	normal_interp = vec3(vertices.data[vertex_idx].normal_xy, vertices.data[vertex_idx].normal_z);

	face_normal = -normalize(cross((vertices.data[vertex_indices.x].position - vertices.data[vertex_indices.y].position), (vertices.data[vertex_indices.x].position - vertices.data[vertex_indices.z].position)));

	v1_uv = vertices.data[vertex_indices.x].uv * params.atlas_size;
	v2_uv = vertices.data[vertex_indices.y].uv * params.atlas_size;
	v3_uv = vertices.data[vertex_indices.z].uv * params.atlas_size;

	v1_pos = vertices.data[vertex_indices.x].position;
	v2_pos = vertices.data[vertex_indices.y].position;
	v3_pos = vertices.data[vertex_indices.z].position;

	{
		const float FLAT_THRESHOLD = 0.99;
		const vec3 norm_a = vec3(vertices.data[vertex_indices.x].normal_xy, vertices.data[vertex_indices.x].normal_z);
		const vec3 norm_b = vec3(vertices.data[vertex_indices.y].normal_xy, vertices.data[vertex_indices.y].normal_z);
		const vec3 norm_c = vec3(vertices.data[vertex_indices.z].normal_xy, vertices.data[vertex_indices.z].normal_z);
		fragment_action = (dot(norm_a, norm_b) < FLAT_THRESHOLD || dot(norm_a, norm_c) < FLAT_THRESHOLD || dot(norm_b, norm_c) < FLAT_THRESHOLD) ? FA_SMOOTHEN_POSITION : FA_NONE;
	}

	gl_Position = vec4((uv_interp + params.uv_offset) * 2.0 - 1.0, 0.0001, 1.0);
}

#[fragment]

#version 450

#VERSION_DEFINES

#include "lm_common_inc.glsl"

layout(push_constant, std430) uniform Params {
	vec2 atlas_size;
	vec2 uv_offset;
	vec3 to_cell_size;
	uint base_triangle;
	vec3 to_cell_offset;
	float bias;
	ivec3 grid_size;
	uint pad2;
}
params;

layout(location = 0) in vec3 vertex_interp;
layout(location = 1) in vec3 normal_interp;
layout(location = 2) in vec2 uv_interp;
layout(location = 3) in vec3 barycentric;
layout(location = 4) in flat uvec3 vertex_indices;
layout(location = 5) in flat vec3 face_normal;
layout(location = 6) in flat uint fragment_action;
layout(location = 7) in flat vec2 v1_uv;
layout(location = 8) in flat vec2 v2_uv;
layout(location = 9) in flat vec2 v3_uv;
layout(location = 10) in flat vec3 v1_pos;
layout(location = 11) in flat vec3 v2_pos;
layout(location = 12) in flat vec3 v3_pos;

layout(location = 0) out vec4 position;
layout(location = 1) out vec4 normal;
layout(location = 2) out vec4 unocclude;

const float EDGE_CORRECTION_DISTANCE = 0.01;

// Simplified cross function for 2d vectors. Return same value as cross(vec3(a, 0.0), vec3(b, 0.0)).z.
float cross2d(const vec2 a, const vec2 b) {
	return a.x * b.y - a.y * b.x;
}

// Same as /core/math/geometry_3d.h's triangle_get_barycentric_coords function except for 2d vectors.
vec3 triangle_get_barycentric_coords(const vec2 p_a, const vec2 p_b, const vec2 p_c, const vec2 p_pos) {
	vec2 v0 = p_b - p_a;
	vec2 v1 = p_c - p_a;
	vec2 v2 = p_pos - p_a;

	float d00 = dot(v0, v0);
	float d01 = dot(v0, v1);
	float d11 = dot(v1, v1);
	float d20 = dot(v2, v0);
	float d21 = dot(v2, v1);
	float denom = (d00 * d11 - d01 * d01);
	if (denom == 0) {
		return vec3(0.0); //invalid triangle, return empty
	}
	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;
	float u = 1.0 - v - w;
	return vec3(u, v, w);
}

// Returns perpendicular direction from triangle edge towards triangle's inside.
vec2 get_cross_dir(const vec2 v1, const vec2 v2) {
	vec2 uv_center = (v1_uv + v2_uv + v3_uv) / 3.0;
	float uv_sign = sign(cross2d(v1 - uv_center, v2 - uv_center));
	return normalize(cross(
			vec3(v2 - v1, 0.0),
			vec3(0.0, 0.0, uv_sign))
					.xy);
}

void main() {
	vec3 vertex_pos = vertex_interp;

	if (fragment_action == FA_SMOOTHEN_POSITION) {
		// smooth out vertex position by interpolating its projection in the 3 normal planes (normal plane is created by vertex pos and normal)
		// because we don't want to interpolate inwards, normals found pointing inwards are pushed out.
		vec3 pos_a = vertices.data[vertex_indices.x].position;
		vec3 pos_b = vertices.data[vertex_indices.y].position;
		vec3 pos_c = vertices.data[vertex_indices.z].position;
		vec3 center = (pos_a + pos_b + pos_c) * 0.3333333;
		vec3 norm_a = vec3(vertices.data[vertex_indices.x].normal_xy, vertices.data[vertex_indices.x].normal_z);
		vec3 norm_b = vec3(vertices.data[vertex_indices.y].normal_xy, vertices.data[vertex_indices.y].normal_z);
		vec3 norm_c = vec3(vertices.data[vertex_indices.z].normal_xy, vertices.data[vertex_indices.z].normal_z);

		{
			vec3 dir_a = normalize(pos_a - center);
			float d_a = dot(dir_a, norm_a);
			if (d_a < 0) {
				//pointing inwards
				norm_a = normalize(norm_a - dir_a * d_a);
			}
		}
		{
			vec3 dir_b = normalize(pos_b - center);
			float d_b = dot(dir_b, norm_b);
			if (d_b < 0) {
				//pointing inwards
				norm_b = normalize(norm_b - dir_b * d_b);
			}
		}
		{
			vec3 dir_c = normalize(pos_c - center);
			float d_c = dot(dir_c, norm_c);
			if (d_c < 0) {
				//pointing inwards
				norm_c = normalize(norm_c - dir_c * d_c);
			}
		}

		float d_a = dot(norm_a, pos_a);
		float d_b = dot(norm_b, pos_b);
		float d_c = dot(norm_c, pos_c);

		vec3 proj_a = vertex_pos - norm_a * (dot(norm_a, vertex_pos) - d_a);
		vec3 proj_b = vertex_pos - norm_b * (dot(norm_b, vertex_pos) - d_b);
		vec3 proj_c = vertex_pos - norm_c * (dot(norm_c, vertex_pos) - d_c);

		vec3 smooth_position = proj_a * barycentric.x + proj_b * barycentric.y + proj_c * barycentric.z;

		if (dot(face_normal, smooth_position) > dot(face_normal, vertex_pos)) { //only project outwards
			vertex_pos = smooth_position;
		}
	}

	{
		// unocclusion technique based on:
		// https://ndotl.wordpress.com/2018/08/29/baking-artifact-free-lightmaps/

		/* compute texel size */
		vec3 delta_uv = max(abs(dFdx(vertex_interp)), abs(dFdy(vertex_interp)));
		float texel_size = max(delta_uv.x, max(delta_uv.y, delta_uv.z));
		texel_size *= sqrt(2.0); //expand to unit box edge length (again, worst case)

		unocclude.xyz = face_normal;
		unocclude.w = texel_size;

		//continued on lm_compute.glsl
	}

	vec2 coords = gl_FragCoord.xy;

	// Get distance in texels from each triangle's edge in UV space. Each edge's distance is stored as a component of the resulting vec3.
	vec3 xyz = vec3(
					   cross2d(normalize(v2_uv - v1_uv), coords - v1_uv),
					   cross2d(normalize(v3_uv - v2_uv), coords - v2_uv),
					   cross2d(normalize(v1_uv - v3_uv), coords - v3_uv)) *
			sign(cross2d(v3_uv - v1_uv, v2_uv - v1_uv));

	vec2 cross_dir = vec2(0.0);

	// Checking per edge, then adding the result. Preferred over just looking for the closest edge to account for cases where a texel in a triangle's corner is too close to two edges.
	if (xyz.x > -EDGE_CORRECTION_DISTANCE) {
		cross_dir += get_cross_dir(v1_uv, v2_uv);
	}
	if (xyz.y > -EDGE_CORRECTION_DISTANCE) {
		cross_dir += get_cross_dir(v2_uv, v3_uv);
	}
	if (xyz.z > -EDGE_CORRECTION_DISTANCE) {
		cross_dir += get_cross_dir(v3_uv, v1_uv);
	}

	if (length(cross_dir) > 0.1) {
		// Get barycentric coordinates from the neighboring texel towards cross_dir.
		vec3 offset_barycentric = triangle_get_barycentric_coords(v1_uv, v2_uv, v3_uv, gl_FragCoord.xy + cross_dir);

		// Get the directional vector towards this texel in 3D space.
		vec3 offset_dir = normalize((v1_pos * offset_barycentric.x + v2_pos * offset_barycentric.y + v3_pos * offset_barycentric.z) - vertex_pos);

		position = vec4(vertex_pos - offset_dir * EDGE_CORRECTION_DISTANCE, 1.0);
	} else {
		position = vec4(vertex_pos, 1.0);
	}

	normal = vec4(normalize(normal_interp), 1.0);

	// Uncomment to no longer keep position values away from triangle edges (for testing only).
	//position = vec4(vertex_pos, 1.0);
}
