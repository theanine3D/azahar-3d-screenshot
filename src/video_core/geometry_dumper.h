// Copyright Citra Emulator Project / Azahar Emulator Project
// Licensed under GPLv2 or any later version
// Refer to the license.txt file included.

#pragma once

#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include "common/common_types.h"
#include "common/vector_math.h"
#include "video_core/pica/regs_texturing.h"

namespace Memory {
class MemorySystem;
}

namespace Pica {
struct OutputVertex;
}

namespace VideoCore {

class GeometryDumper {
public:
    struct TextureKey {
        PAddr address = 0;
        u32 width = 0;
        u32 height = 0;
        Pica::TexturingRegs::TextureFormat format{};

        bool IsValid() const { return address != 0 && width > 0 && height > 0; }

        bool operator<(const TextureKey& o) const {
            if (address != o.address) return address < o.address;
            if (width != o.width) return width < o.width;
            if (height != o.height) return height < o.height;
            return format < o.format;
        }
        bool operator==(const TextureKey& o) const {
            return address == o.address && width == o.width && height == o.height &&
                   format == o.format;
        }
        bool operator!=(const TextureKey& o) const { return !(*this == o); }
    };

    struct CapturedVertex {
        Common::Vec3f position;  // clip-space derived: (W, X, -Y)
        Common::Vec3f view_pos;  // view-space position from vertex shader (if available)
        Common::Vec4f color;
        Common::Vec2f tc[3];
    };

    struct CapturedMesh {
        std::vector<CapturedVertex> vertices;
        std::array<TextureKey, 3> tex_keys{};
        bool has_tc[3]{};
        bool has_view_pos = false; // true when the game outputs view-space positions
    };

    void RequestCapture(const std::string& output_dir);
    bool IsCapturing() const { return capturing; }

    void BeginFrame();
    void RecordTriangle(const Pica::OutputVertex& v0, const Pica::OutputVertex& v1,
                        const Pica::OutputVertex& v2,
                        const Pica::TexturingRegs& texturing);
    void FlushBatch();
    void EndFrame(Memory::MemorySystem& memory);

private:
    std::atomic_bool capture_requested{false};
    bool capturing = false;
    std::string pending_output_dir;
    std::string output_dir;

    std::vector<CapturedVertex> current_batch;
    std::array<TextureKey, 3> current_tex_keys{};
    bool current_has_tc[3]{};

    std::vector<CapturedMesh> frame_meshes;
    // Map from TextureKey array (as tuple key via map<>) to mesh index
    std::map<std::array<TextureKey, 3>, std::size_t> mesh_index_map;

    static TextureKey MakeKey(const Pica::TexturingRegs::FullTextureConfig& tex);
    void CommitCurrentBatch();
    void SaveFrame(Memory::MemorySystem& memory);
    static std::vector<u8> DecodeTexture(Memory::MemorySystem& memory, const TextureKey& key);
    static void WriteGltf(const std::vector<CapturedMesh>& meshes,
                          const std::map<TextureKey, std::string>& tex_files,
                          const std::string& dir);
};

} // namespace VideoCore
