// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <fmt/format.h>
#include <lodepng.h>
#include "common/file_util.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "video_core/geometry_dumper.h"
#include "video_core/pica/output_vertex.h"
#include "video_core/texture/texture_decode.h"

namespace VideoCore {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------


GeometryDumper::TextureKey GeometryDumper::MakeKey(
    const Pica::TexturingRegs::FullTextureConfig& tex) {
    if (!tex.enabled) {
        return {};
    }
    const PAddr addr = tex.config.GetPhysicalAddress();
    if (addr == 0) {
        return {};
    }
    TextureKey k;
    k.address = addr;
    k.width = tex.config.width;
    k.height = tex.config.height;
    k.format = tex.format;
    return k;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void GeometryDumper::RequestCapture(const std::string& output_dir) {
    pending_output_dir = output_dir;
    capture_requested.store(true, std::memory_order_release);
}

void GeometryDumper::BeginFrame() {
    if (capture_requested.exchange(false, std::memory_order_acquire)) {
        capturing = true;
        output_dir = pending_output_dir;
        frame_meshes.clear();
        mesh_index_map.clear();
        current_batch.clear();
        std::fill(std::begin(current_has_tc), std::end(current_has_tc), false);
        current_tex_keys = {};
    }
}

void GeometryDumper::RecordTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                                    const Pica::OutputVertex& v2,
                                    const Pica::TexturingRegs& texturing) {
    const auto textures = texturing.GetTextures();

    // Check if texture state changed since last batch
    std::array<TextureKey, 3> new_keys{
        MakeKey(textures[0]),
        MakeKey(textures[1]),
        MakeKey(textures[2]),
    };

    if (!current_batch.empty() && new_keys != current_tex_keys) {
        CommitCurrentBatch();
    }

    current_tex_keys = new_keys;

    // Record UV presence
    for (int i = 0; i < 3; ++i) {
        current_has_tc[i] = current_has_tc[i] || new_keys[i].IsValid();
    }

    // Swap v1/v2 to reverse winding order so faces are front-facing (CCW) in glTF/Blender.
    const Pica::OutputVertex* verts[3] = {&v0, &v2, &v1};
    for (const auto* v : verts) {
        CapturedVertex cv;

        // Map clip-space to glTF-oriented world space.
        // Raw clip: X=screen-right, Y=screen-up, W=view-depth (positive forward).
        // glTF expects Y-up, -Z forward (right-handed). Two corrections applied:
        //   1. Negate W (glTF depth is negative-forward): (X, Y, -W)
        //   2. Rotate -90° around Y: (W, Y, X)
        // Combined: X=W_clip, Y=Y_clip, Z=X_clip — bakes out the manual
        // Blender corrections the user would otherwise need to apply after import.
        cv.position = {
            v->pos.w.ToFloat32(),    // W_clip  → glTF X
            v->pos.x.ToFloat32(),    // X_clip  → glTF Y
           -v->pos.y.ToFloat32(),    // -Y_clip → glTF Z
        };

        cv.view_pos = {
            v->view.x.ToFloat32(),
            v->view.y.ToFloat32(),
            v->view.z.ToFloat32(),
        };

        cv.color = {
            v->color.x.ToFloat32(),
            v->color.y.ToFloat32(),
            v->color.z.ToFloat32(),
            v->color.w.ToFloat32(),
        };

        cv.tc[0] = {v->tc0.x.ToFloat32(), v->tc0.y.ToFloat32()};
        cv.tc[1] = {v->tc1.x.ToFloat32(), v->tc1.y.ToFloat32()};
        cv.tc[2] = {v->tc2.x.ToFloat32(), v->tc2.y.ToFloat32()};

        current_batch.push_back(cv);
    }
}

void GeometryDumper::FlushBatch() {
    if (!capturing || current_batch.empty()) {
        return;
    }
    CommitCurrentBatch();
}

void GeometryDumper::EndFrame(Memory::MemorySystem& memory) {
    if (!capturing) {
        return;
    }
    capturing = false;

    // Commit any remaining batch
    if (!current_batch.empty()) {
        CommitCurrentBatch();
    }

    if (frame_meshes.empty()) {
        LOG_WARNING(Render, "3D capture: no geometry captured");
        return;
    }

    SaveFrame(memory);
}

// ---------------------------------------------------------------------------
// Private: batch management
// ---------------------------------------------------------------------------

void GeometryDumper::CommitCurrentBatch() {
    if (current_batch.empty()) {
        return;
    }

    // Detect whether this batch has view-space positions (game populated the VIEW register).
    // Any non-zero view_pos in the batch is sufficient — a vertex exactly at the camera
    // origin is essentially impossible in practice.
    const bool batch_has_view = std::any_of(current_batch.begin(), current_batch.end(),
        [](const CapturedVertex& v) {
            return v.view_pos.x != 0.0f || v.view_pos.y != 0.0f || v.view_pos.z != 0.0f;
        });

    auto it = mesh_index_map.find(current_tex_keys);
    if (it == mesh_index_map.end()) {
        CapturedMesh mesh;
        mesh.tex_keys = current_tex_keys;
        std::copy(std::begin(current_has_tc), std::end(current_has_tc),
                  std::begin(mesh.has_tc));
        mesh.has_view_pos = batch_has_view;
        mesh.vertices = std::move(current_batch);
        mesh_index_map[current_tex_keys] = frame_meshes.size();
        frame_meshes.push_back(std::move(mesh));
    } else {
        CapturedMesh& mesh = frame_meshes[it->second];
        for (int i = 0; i < 3; ++i) {
            mesh.has_tc[i] = mesh.has_tc[i] || current_has_tc[i];
        }
        mesh.has_view_pos = mesh.has_view_pos || batch_has_view;
        mesh.vertices.insert(mesh.vertices.end(), current_batch.begin(), current_batch.end());
    }

    current_batch.clear();
    std::fill(std::begin(current_has_tc), std::end(current_has_tc), false);
    current_tex_keys = {};
}

// ---------------------------------------------------------------------------
// Private: texture decoding
// ---------------------------------------------------------------------------

std::vector<u8> GeometryDumper::DecodeTexture(Memory::MemorySystem& memory,
                                              const TextureKey& key) {
    const u8* src = memory.GetPhysicalPointer(key.address);
    if (!src) {
        return {};
    }

    Pica::Texture::TextureInfo info;
    info.physical_address = key.address;
    info.width = key.width;
    info.height = key.height;
    info.format = key.format;
    info.SetDefaultStride();

    std::vector<u8> rgba(key.width * key.height * 4);
    for (u32 y = 0; y < key.height; ++y) {
        for (u32 x = 0; x < key.width; ++x) {
            const auto texel = Pica::Texture::LookupTexture(src, x, y, info);
            const u32 idx = (y * key.width + x) * 4;
            rgba[idx + 0] = texel.x;
            rgba[idx + 1] = texel.y;
            rgba[idx + 2] = texel.z;
            rgba[idx + 3] = texel.w;
        }
    }
    return rgba;
}

// ---------------------------------------------------------------------------
// Private: save frame
// ---------------------------------------------------------------------------

void GeometryDumper::SaveFrame(Memory::MemorySystem& memory) {
    // Create output directory
    if (!FileUtil::CreateFullPath(output_dir + "/")) {
        LOG_ERROR(Render, "3D capture: failed to create output directory '{}'", output_dir);
        return;
    }

    // Collect unique textures and save as PNG
    std::map<TextureKey, std::string> tex_files; // key -> filename
    std::map<TextureKey, int> tex_index;
    int tex_counter = 0;

    for (const auto& mesh : frame_meshes) {
        for (int i = 0; i < 3; ++i) {
            const auto& key = mesh.tex_keys[i];
            if (!key.IsValid() || tex_files.count(key)) {
                continue;
            }
            const std::string filename = fmt::format("tex_{}.png", tex_counter++);
            const std::string full_path = output_dir + "/" + filename;

            auto rgba = DecodeTexture(memory, key);
            if (rgba.empty()) {
                LOG_WARNING(Render, "3D capture: failed to decode texture at 0x{:08X}", key.address);
                continue;
            }

            const u32 err = lodepng::encode(full_path, rgba.data(), key.width, key.height);
            if (err) {
                LOG_ERROR(Render, "3D capture: PNG encode error {}: {}", err,
                          lodepng_error_text(err));
            } else {
                tex_files[key] = filename;
                tex_index[key] = static_cast<int>(tex_files.size()) - 1;
            }
        }
    }

    WriteGltf(frame_meshes, tex_files, output_dir);
    LOG_INFO(Render, "3D capture saved to '{}'", output_dir);
}

// ---------------------------------------------------------------------------
// Private: glTF 2.0 writer
// ---------------------------------------------------------------------------

namespace {

// Append raw bytes to a buffer and return the starting offset.
template <typename T>
static u32 AppendData(std::vector<u8>& buf, const T* data, std::size_t count) {
    const u32 offset = static_cast<u32>(buf.size());
    const u8* raw = reinterpret_cast<const u8*>(data);
    buf.insert(buf.end(), raw, raw + sizeof(T) * count);
    return offset;
}

// Pad buffer to 4-byte alignment.
static void Align4(std::vector<u8>& buf) {
    while (buf.size() % 4 != 0) {
        buf.push_back(0);
    }
}

} // anonymous namespace

void GeometryDumper::WriteGltf(const std::vector<CapturedMesh>& meshes,
                               const std::map<TextureKey, std::string>& tex_files,
                               const std::string& dir) {
    // Build a stable ordered list of (TextureKey -> glTF image index)
    std::map<TextureKey, int> image_index;
    {
        int idx = 0;
        for (const auto& [key, filename] : tex_files) {
            image_index[key] = idx++;
        }
    }

    // -----------------------------------------------------------------------
    // Binary buffer: pack all mesh attribute data
    // -----------------------------------------------------------------------
    std::vector<u8> bin;

    // Per-mesh accessor info
    struct MeshAccessors {
        u32 pos_bv, norm_bv, col_bv, tc_bv[3];
        u32 pos_offset, col_offset, tc_offset[3];
        u32 vertex_count;
        bool has_tc[3];
    };
    std::vector<MeshAccessors> mesh_acc(meshes.size());

    struct BufferViewInfo {
        u32 byte_offset;
        u32 byte_length;
        int target; // 34962 = ARRAY_BUFFER
    };
    std::vector<BufferViewInfo> bv_list;

    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const auto& mesh = meshes[mi];
        auto& acc = mesh_acc[mi];
        acc.vertex_count = static_cast<u32>(mesh.vertices.size());
        std::copy(std::begin(mesh.has_tc), std::end(mesh.has_tc), std::begin(acc.has_tc));

        Align4(bin);

        // POSITION: Vec3f
        // Prefer view-space (view_pos) when the game outputs it — no perspective
        // distortion. Fall back to clip-space derived position otherwise.
        {
            std::vector<float> pos_data;
            pos_data.reserve(mesh.vertices.size() * 3);
            if (mesh.has_view_pos) {
                // Negate Y only; winding swap (v0,v2,v1) corrects handedness.
                for (const auto& v : mesh.vertices) {
                    pos_data.push_back(v.view_pos.x);
                    pos_data.push_back(-v.view_pos.y);
                    pos_data.push_back(v.view_pos.z);
                }
            } else {
                for (const auto& v : mesh.vertices) {
                    pos_data.push_back(v.position.x);
                    pos_data.push_back(v.position.y);
                    pos_data.push_back(v.position.z);
                }
            }
            const u32 byte_offset = AppendData(bin, pos_data.data(), pos_data.size());
            const u32 byte_length = static_cast<u32>(pos_data.size() * sizeof(float));
            acc.pos_bv = static_cast<u32>(bv_list.size());
            acc.pos_offset = byte_offset;
            bv_list.push_back({byte_offset, byte_length, 34962});
            Align4(bin);
        }

        // NORMAL: Vec3f (flat normals computed from triangle edges)
        {
            std::vector<float> norm_data;
            norm_data.reserve(mesh.vertices.size() * 3);
            const std::size_t tri_count = mesh.vertices.size() / 3;
            for (std::size_t ti = 0; ti < tri_count; ++ti) {
                Common::Vec3f p0, p1, p2;
                if (mesh.has_view_pos) {
                    // Use the same transform applied to exported positions (-X, -Y, Z)
                    // so the cross product is consistent with the actual winding in the file.
                    auto xform = [](const Common::Vec3f& v) -> Common::Vec3f {
                        return {v.x, -v.y, v.z};
                    };
                    p0 = xform(mesh.vertices[ti * 3 + 0].view_pos);
                    p1 = xform(mesh.vertices[ti * 3 + 1].view_pos);
                    p2 = xform(mesh.vertices[ti * 3 + 2].view_pos);
                } else {
                    p0 = mesh.vertices[ti * 3 + 0].position;
                    p1 = mesh.vertices[ti * 3 + 1].position;
                    p2 = mesh.vertices[ti * 3 + 2].position;
                }
                const float ex1 = p1.x - p0.x, ey1 = p1.y - p0.y, ez1 = p1.z - p0.z;
                const float ex2 = p2.x - p0.x, ey2 = p2.y - p0.y, ez2 = p2.z - p0.z;
                float nx = ey1 * ez2 - ez1 * ey2;
                float ny = ez1 * ex2 - ex1 * ez2;
                float nz = ex1 * ey2 - ey1 * ex2;
                const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (mesh.has_view_pos) {
                    if (len > 1e-6f) { nx /= len; ny /= len; nz /= len; }
                } else {
                    // Clip-space remap reverses winding — negate to fix.
                    if (len > 1e-6f) { nx /= -len; ny /= -len; nz /= -len; }
                }
                for (int k = 0; k < 3; ++k) {
                    norm_data.push_back(nx);
                    norm_data.push_back(ny);
                    norm_data.push_back(nz);
                }
            }
            const u32 byte_offset = AppendData(bin, norm_data.data(), norm_data.size());
            const u32 byte_length = static_cast<u32>(norm_data.size() * sizeof(float));
            acc.norm_bv = static_cast<u32>(bv_list.size());
            bv_list.push_back({byte_offset, byte_length, 34962});
            Align4(bin);
        }

        // COLOR_0: Vec4f
        {
            std::vector<float> col_data;
            col_data.reserve(mesh.vertices.size() * 4);
            for (const auto& v : mesh.vertices) {
                col_data.push_back(v.color.x);
                col_data.push_back(v.color.y);
                col_data.push_back(v.color.z);
                col_data.push_back(v.color.w);
            }
            const u32 byte_offset = AppendData(bin, col_data.data(), col_data.size());
            const u32 byte_length = static_cast<u32>(col_data.size() * sizeof(float));
            acc.col_bv = static_cast<u32>(bv_list.size());
            acc.col_offset = byte_offset;
            bv_list.push_back({byte_offset, byte_length, 34962});
            Align4(bin);
        }

        // TEXCOORD_0/1/2
        for (int ti = 0; ti < 3; ++ti) {
            if (!mesh.has_tc[ti]) {
                acc.tc_bv[ti] = 0;
                acc.tc_offset[ti] = 0;
                continue;
            }
            std::vector<float> tc_data;
            tc_data.reserve(mesh.vertices.size() * 2);
            for (const auto& v : mesh.vertices) {
                tc_data.push_back(v.tc[ti].x);
                tc_data.push_back(1.0f - v.tc[ti].y);
            }
            const u32 byte_offset = AppendData(bin, tc_data.data(), tc_data.size());
            const u32 byte_length = static_cast<u32>(tc_data.size() * sizeof(float));
            acc.tc_bv[ti] = static_cast<u32>(bv_list.size());
            acc.tc_offset[ti] = byte_offset;
            bv_list.push_back({byte_offset, byte_length, 34962});
            Align4(bin);
        }
    }

    // Write .bin
    const std::string bin_path = dir + "/scene.bin";
    {
        std::ofstream f(bin_path, std::ios::binary);
        if (!f) {
            LOG_ERROR(Render, "3D capture: failed to open '{}' for writing", bin_path);
            return;
        }
        f.write(reinterpret_cast<const char*>(bin.data()),
                static_cast<std::streamsize>(bin.size()));
    }

    // -----------------------------------------------------------------------
    // JSON assembly
    // -----------------------------------------------------------------------
    std::ostringstream j;
    j << "{\n";
    j << "  \"asset\": {\"version\": \"2.0\", \"generator\": \"Azahar 3D Screenshot\"},\n";
    j << "  \"scene\": 0,\n";

    // scenes + nodes
    j << "  \"scenes\": [{\"nodes\": [";
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        if (mi > 0) j << ", ";
        j << mi;
    }
    j << "]}],\n";

    j << "  \"nodes\": [\n";
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        j << "    {\"mesh\": " << mi << "}";
        if (mi + 1 < meshes.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";

    // materials
    // Build: for each unique material (unique tex_keys triple) a material entry
    // We'll have one material per mesh (already deduplicated via mesh_index_map)
    j << "  \"materials\": [\n";
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const auto& mesh = meshes[mi];
        j << "    {\n";
        j << "      \"name\": \"mat_" << mi << "\",\n";
        j << "      \"pbrMetallicRoughness\": {\n";
        j << "        \"metallicFactor\": 0.0,\n";
        j << "        \"roughnessFactor\": 1.0";

        // baseColorTexture from slot 0
        if (mesh.tex_keys[0].IsValid() && image_index.count(mesh.tex_keys[0])) {
            j << ",\n        \"baseColorTexture\": {\"index\": "
              << image_index.at(mesh.tex_keys[0]) << "}";
        }
        j << "\n      }";

        // occlusionTexture from slot 1 (preserves the texture data)
        if (mesh.tex_keys[1].IsValid() && image_index.count(mesh.tex_keys[1])) {
            j << ",\n      \"occlusionTexture\": {\"index\": "
              << image_index.at(mesh.tex_keys[1]) << "}";
        }

        // emissiveTexture from slot 2
        if (mesh.tex_keys[2].IsValid() && image_index.count(mesh.tex_keys[2])) {
            j << ",\n      \"emissiveTexture\": {\"index\": "
              << image_index.at(mesh.tex_keys[2]) << "}";
        }

        j << ",\n      \"doubleSided\": true";
        j << "\n    }";
        if (mi + 1 < meshes.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";

    // meshes
    int accessor_idx = 0;
    j << "  \"meshes\": [\n";
    for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
        const auto& acc = mesh_acc[mi];
        j << "    {\n";
        j << "      \"name\": \"mesh_" << mi << "\",\n";
        j << "      \"primitives\": [{\n";
        j << "        \"attributes\": {\n";
        j << "          \"POSITION\": " << (accessor_idx) << ",\n";
        j << "          \"NORMAL\": " << (accessor_idx + 1) << ",\n";
        j << "          \"COLOR_0\": " << (accessor_idx + 2);

        int tc_acc_offset = accessor_idx + 3;
        for (int ti = 0; ti < 3; ++ti) {
            if (acc.has_tc[ti]) {
                j << ",\n          \"TEXCOORD_" << ti << "\": " << tc_acc_offset++;
            }
        }
        j << "\n        },\n";
        j << "        \"material\": " << mi << "\n";
        j << "      }]\n";
        j << "    }";
        if (mi + 1 < meshes.size()) j << ",";
        j << "\n";

        // Advance accessor index: pos + norm + col + active tc channels
        accessor_idx += 3;
        for (int ti = 0; ti < 3; ++ti) {
            if (acc.has_tc[ti]) ++accessor_idx;
        }
    }
    j << "  ],\n";

    // accessors
    j << "  \"accessors\": [\n";
    {
        bool first = true;
        // We need to iterate meshes in order to emit accessors in the same order
        // as accessor_idx above. Reset and redo.
        int bv_idx = 0;
        for (std::size_t mi = 0; mi < meshes.size(); ++mi) {
            const auto& acc = mesh_acc[mi];
            const u32 vc = acc.vertex_count;

            auto emit = [&](const char* type, int comp_type, u32 bv_i) {
                if (!first) j << ",\n";
                first = false;
                j << "    {\"bufferView\": " << bv_i
                  << ", \"byteOffset\": 0"
                  << ", \"componentType\": " << comp_type
                  << ", \"count\": " << vc
                  << ", \"type\": \"" << type << "\"}";
            };

            emit("VEC3", 5126, bv_idx); ++bv_idx; // POSITION
            emit("VEC3", 5126, bv_idx); ++bv_idx; // NORMAL
            emit("VEC4", 5126, bv_idx); ++bv_idx; // COLOR_0
            for (int ti = 0; ti < 3; ++ti) {
                if (acc.has_tc[ti]) {
                    emit("VEC2", 5126, bv_idx);
                    ++bv_idx;
                }
            }
        }
    }
    j << "\n  ],\n";

    // bufferViews
    j << "  \"bufferViews\": [\n";
    for (std::size_t bvi = 0; bvi < bv_list.size(); ++bvi) {
        const auto& bv = bv_list[bvi];
        j << "    {\"buffer\": 0, \"byteOffset\": " << bv.byte_offset
          << ", \"byteLength\": " << bv.byte_length
          << ", \"target\": " << bv.target << "}";
        if (bvi + 1 < bv_list.size()) j << ",";
        j << "\n";
    }
    j << "  ],\n";

    // buffers
    j << "  \"buffers\": [{\"uri\": \"scene.bin\", \"byteLength\": " << bin.size() << "}],\n";

    // images
    j << "  \"images\": [\n";
    {
        bool first = true;
        for (const auto& [key, filename] : tex_files) {
            if (!first) j << ",\n";
            first = false;
            j << "    {\"uri\": \"" << filename << "\"}";
        }
    }
    j << "\n  ],\n";

    // textures (one per image, same order)
    j << "  \"textures\": [\n";
    {
        bool first = true;
        int idx = 0;
        for (const auto& [key, filename] : tex_files) {
            if (!first) j << ",\n";
            first = false;
            j << "    {\"source\": " << idx++ << "}";
        }
    }
    j << "\n  ]\n";

    j << "}\n";

    const std::string gltf_path = dir + "/scene.gltf";
    std::ofstream gf(gltf_path);
    if (!gf) {
        LOG_ERROR(Render, "3D capture: failed to write '{}'", gltf_path);
        return;
    }
    gf << j.str();
}

} // namespace VideoCore
