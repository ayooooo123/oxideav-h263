//! The registry decoder reports the size of the frame it last returned
//! (oxideav-core `Decoder::output_video_dimensions` /
//! `output_pixel_format`), including size changes mid-stream: sub-QCIF,
//! then QCIF, then a 48×32 PLUSPTYPE custom format.
//!
//! `decoded_format/*.h263` are three-picture FFmpeg encodes of `testsrc`
//! (`h263`, `h263` and `h263p`); FFmpeg byte-aligns every picture start
//! code.

use oxideav_core::{
    CodecId, CodecParameters, Decoder, Error, Frame, Packet, PixelFormat, TimeBase, VideoFrame,
};

fn stream() -> Vec<u8> {
    [
        &include_bytes!("decoded_format/a_sqcif.h263")[..],
        &include_bytes!("decoded_format/b_qcif.h263")[..],
        &include_bytes!("decoded_format/c_custom.h263")[..],
    ]
    .concat()
}

const EXPECTED: [(u32, u32); 9] = [
    (128, 96),
    (128, 96),
    (128, 96),
    (176, 144),
    (176, 144),
    (176, 144),
    (48, 32),
    (48, 32),
    (48, 32),
];

fn decoder() -> Box<dyn Decoder> {
    oxideav_h263::codec::make_decoder(&CodecParameters::video(CodecId::new("h263")))
        .expect("decoder")
}

fn report(dec: &dyn Decoder) -> Option<(u32, u32, PixelFormat)> {
    let (w, h) = dec.output_video_dimensions()?;
    Some((w, h, dec.output_pixel_format()?))
}

fn assert_geometry(frame: &VideoFrame, (w, h): (u32, u32), at: usize) {
    let planes = frame.image_planes();
    assert_eq!(planes.len(), 3, "frame {at}: planes");
    for (i, plane) in planes.iter().enumerate() {
        let (pw, ph) = if i == 0 { (w, h) } else { (w / 2, h / 2) };
        assert_eq!(plane.stride, pw as usize, "frame {at}: plane {i} stride");
        assert_eq!(
            plane.data.len(),
            plane.stride * ph as usize,
            "frame {at}: plane {i} rows"
        );
    }
}

/// Receives every available frame, checking each right after it is
/// returned.
fn receive_all(dec: &mut dyn Decoder, seen: &mut usize) {
    loop {
        match dec.receive_frame() {
            Ok(Frame::Video(frame)) => {
                let (w, h) = EXPECTED[*seen];
                assert_eq!(
                    report(dec),
                    Some((w, h, PixelFormat::Yuv420P)),
                    "report after frame {seen}"
                );
                assert_geometry(&frame, (w, h), *seen);
                *seen += 1;
            }
            Ok(_) => panic!("non-video frame"),
            Err(Error::NeedMore | Error::Eof) => return,
            Err(e) => panic!("receive: {e}"),
        }
    }
}

/// All three streams in one packet: later sizes are decoded while the
/// first stream's frames still wait to be returned.
#[test]
fn each_frame_reports_its_own_size() {
    let mut dec = decoder();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 25), stream()))
        .expect("send");
    assert_eq!(
        report(&*dec),
        Some((128, 96, PixelFormat::Yuv420P)),
        "report before the first frame"
    );
    let mut seen = 0;
    receive_all(&mut *dec, &mut seen);
    dec.flush().expect("flush");
    receive_all(&mut *dec, &mut seen);
    assert_eq!(seen, EXPECTED.len(), "frames");
    assert_eq!(report(&*dec), Some((48, 32, PixelFormat::Yuv420P)));
}

/// One picture per packet, receiving after each.
#[test]
fn reports_change_with_the_frame_that_carries_the_change() {
    let stream = stream();
    let starts: Vec<usize> = (0..stream.len() - 2)
        .filter(|&i| stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] >> 2 == 0x20)
        .collect();
    assert_eq!(starts.len(), EXPECTED.len(), "picture start codes");
    let mut dec = decoder();
    let mut seen = 0;
    for (k, &start) in starts.iter().enumerate() {
        let end = starts.get(k + 1).copied().unwrap_or(stream.len());
        dec.send_packet(&Packet::new(
            0,
            TimeBase::new(1, 25),
            stream[start..end].to_vec(),
        ))
        .expect("send");
        receive_all(&mut *dec, &mut seen);
    }
    dec.flush().expect("flush");
    receive_all(&mut *dec, &mut seen);
    assert_eq!(seen, EXPECTED.len(), "frames");
}
