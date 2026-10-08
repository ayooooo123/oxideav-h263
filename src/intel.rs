// Intel H.263 picture header.
//
// Ported from FFmpeg libavcodec/intelh263dec.c (commit 2da55bf),
// Copyright (c) 2003 Michael Niedermayer. This file is licensed under the
// GNU Lesser General Public License version 2.1 or later
// (LGPL-2.1-or-later), as the file it is ported from; the rest of this
// crate is MIT.

//! Intel H.263 (`h263i`, AVI FourCC `I263`): H.263 version 1 macroblock
//! layers under a picture header of Intel's own. Source formats 1-5 read
//! as in §5.1.3; format 7 is Intel's extension (loop filter, an Intel
//! PB-frame variant, a custom format whose size is the container's), and
//! formats 0 and 6 are refused, as FFmpeg's `ff_intel_h263_decode_picture_header`
//! does. A packet of exactly 64 bits is a dummy frame that decodes to
//! nothing; the stream decoder drops it.

use oxideav_core::bits::BitReader;

use crate::pb_layer::ModbPresence;
use crate::picture_header::{
    H263PictureCodingType, H263PictureHeader, H263SourceFormat, PSC_BITS, PSC_VALUE,
};
use crate::{Error, Result};

/// The size of a packet FFmpeg skips as a dummy frame
/// (`get_bits_left(&h->gb) == 64`).
pub const DUMMY_FRAME_BYTES: usize = 8;

/// One parsed Intel H.263 picture header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntelPictureHeader {
    /// The baseline-equivalent fields: TR, PTYPE bits 3-5, coding type,
    /// long vectors (Annex D, version 1 form), advanced prediction, and
    /// whether the picture is a PB-frame. SAC is always off.
    pub header: H263PictureHeader,
    /// The luma size the picture decodes at: the source format's for
    /// formats 1-5, otherwise the size in force (FFmpeg reads but skips
    /// the custom format's display size and keeps the size it had,
    /// initially the container's).
    pub size: (u32, u32),
    /// `loop_filter`, set only by the format 7 extension.
    pub loop_filter: bool,
    /// `pb_frame`: 0 none, 1 an Annex G PB-frame, 2 Intel's PB variant
    /// (format 7 extension), whose MODB carries an extra MVDB bit.
    pub pb_frame: u8,
    /// PQUANT, `1..=31`.
    pub pquant: u8,
    /// TRB, for a PB-frame.
    pub trb: u8,
    /// DBQUANT, for a PB-frame.
    pub dbquant: u8,
}

fn bits(reader: &mut BitReader<'_>, n: u32) -> Result<u32> {
    reader.read_u32(n).map_err(|_| Error::UnexpectedEof)
}

fn bit(reader: &mut BitReader<'_>) -> Result<bool> {
    reader.read_bit().map_err(|_| Error::UnexpectedEof)
}

/// `ff_intel_h263_decode_picture_header`, from the Picture Start Code to
/// the first bit of GOB 0. `size_in_force` is the luma size FFmpeg's
/// context holds: the container's until a picture of a standard format
/// sets one.
pub fn parse_intel_picture_header(
    reader: &mut BitReader<'_>,
    size_in_force: (u32, u32),
) -> Result<IntelPictureHeader> {
    if bits(reader, PSC_BITS)? != PSC_VALUE {
        return Err(Error::BadPictureStartCode);
    }
    let temporal_reference = bits(reader, 8)? as u8;
    // `check_marker` after the picture number, then the H.263 id bit.
    if !bit(reader)? || bit(reader)? {
        return Err(Error::BadPtypeFixedBits);
    }
    let split_screen = bit(reader)?;
    let document_camera = bit(reader)?;
    let freeze_release = bit(reader)?;

    let mut format = bits(reader, 3)?;
    if format == 0 || format == 6 {
        // "Intel H.263 free format not supported"
        return Err(Error::ForbiddenSourceFormat);
    }
    let coding_type = if bit(reader)? {
        H263PictureCodingType::Inter
    } else {
        H263PictureCodingType::Intra
    };
    let long_vectors = bit(reader)?;
    if bit(reader)? {
        // "SAC not supported"
        return Err(Error::NotImplemented);
    }
    let obmc = bit(reader)?;
    let mut pb_frame = u8::from(bit(reader)?);
    let mut loop_filter = false;

    let (source_format, size) = if format < 6 {
        let source_format = match format {
            1 => H263SourceFormat::SubQcif,
            2 => H263SourceFormat::Qcif,
            3 => H263SourceFormat::Cif,
            4 => H263SourceFormat::Cif4,
            _ => H263SourceFormat::Cif16,
        };
        let size = source_format.luma_dimensions().ok_or(Error::ForbiddenSourceFormat)?;
        (source_format, size)
    } else {
        format = bits(reader, 3)?;
        if format == 0 || format == 7 {
            // "Wrong Intel H.263 format"
            return Err(Error::ForbiddenSourceFormat);
        }
        // Reserved bits FFmpeg only logs about.
        reader.skip(2).map_err(|_| Error::UnexpectedEof)?;
        loop_filter = bit(reader)?;
        reader.skip(1).map_err(|_| Error::UnexpectedEof)?;
        if bit(reader)? {
            pb_frame = 2;
        }
        reader.skip(5 + 5).map_err(|_| Error::UnexpectedEof)?;
        (H263SourceFormat::Reserved110, size_in_force)
    };
    if format == 6 {
        let aspect_ratio = bits(reader, 4)?;
        // Display width, a marker FFmpeg only logs about, display height.
        reader.skip(9 + 1 + 9).map_err(|_| Error::UnexpectedEof)?;
        if aspect_ratio == 15 {
            reader.skip(8 + 8).map_err(|_| Error::UnexpectedEof)?;
        }
    }

    let pquant = bits(reader, 5)? as u8;
    if pquant == 0 {
        return Err(Error::InvalidQuantiser);
    }
    // Continuous Presence Multipoint: skipped.
    reader.skip(1).map_err(|_| Error::UnexpectedEof)?;
    let (mut trb, mut dbquant) = (0, 0);
    if pb_frame != 0 {
        trb = bits(reader, 3)? as u8;
        dbquant = bits(reader, 2)? as u8;
    }
    // PEI / PSUPP: `skip_1stop_8data_bits`.
    if reader.bits_remaining() == 0 {
        return Err(Error::UnexpectedEof);
    }
    while bit(reader)? {
        reader.skip(8).map_err(|_| Error::UnexpectedEof)?;
        if reader.bits_remaining() == 0 {
            return Err(Error::UnexpectedEof);
        }
    }

    Ok(IntelPictureHeader {
        header: H263PictureHeader {
            temporal_reference,
            split_screen,
            document_camera,
            freeze_release,
            source_format,
            coding_type,
            umv_mode: long_vectors,
            sac_mode: false,
            advanced_prediction: obmc,
            pb_frames: pb_frame != 0,
        },
        size,
        loop_filter,
        pb_frame,
        pquant,
        trb,
        dbquant,
    })
}

/// MODB of Intel's PB-frame variant (`pb_frame` 2): FFmpeg's first MODB
/// bit and `h263_get_modb`. `0` is no B data, `10` MVDB, and after `11`
/// one more bit: `0` CBPB and MVDB, `1` CBPB without MVDB.
pub fn parse_intel_modb(reader: &mut BitReader<'_>) -> Result<ModbPresence> {
    if !bit(reader)? {
        return Ok(ModbPresence::None);
    }
    if !bit(reader)? {
        return Ok(ModbPresence::MvdbOnly);
    }
    Ok(if bit(reader)? { ModbPresence::CbpbOnly } else { ModbPresence::CbpbAndMvdb })
}
