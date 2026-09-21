/* HAM10000 实验报告 docx 生成器（内容来自 report_build/content.json）
 * 结构：R1 封面（DM-1 配色）→ 前置节（摘要+目录，罗马页码）→ 正文节（阿拉伯页码）
 */
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  SectionType, TableOfContents, TableLayoutType, LevelFormat,
} = require("docx");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");

const ROOT = path.resolve(__dirname, "..");
const C = JSON.parse(fs.readFileSync(path.join(__dirname, "content.json"), "utf8"));

// DM-1 Deep Cyan（design-system.md）
const PAL = {
  bg: "162235", accent: "37DCF2",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "1B6B7A", headerText: "FFFFFF", innerLine: "C8DDE2", surface: "EDF3F5" },
};

const allNoBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
  insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
};
const noBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
};

// ---------- design-system.md 封面工具函数（原样实现） ----------
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([..."，。、；：！？", ..."的与和及之在于为", ..."-_—–·/", ..." \t"]);
  const lines = [];
  let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--) {
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++) {
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
      }
    }
    if (breakAt === -1) {
      breakAt = charsPerLine;
      const prevChar = remaining[breakAt - 1], nextChar = remaining[breakAt];
      if (prevChar && nextChar && !breakAfter.has(prevChar) && !breakAfter.has(nextChar) &&
          /[\u4e00-\u9fff]/.test(prevChar) && /[\u4e00-\u9fff]/.test(nextChar)) breakAt -= 1;
    }
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop();
    lines[lines.length - 1] += last;
  }
  return lines;
}

function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / (pt * 20));
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    lines = splitTitleLines(title, charsPerLine(minPt));
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

function calcCoverSpacing(params) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838, marginTop = 0, marginBottom = 0 } = params;
  const SAFETY = 1200;
  const usableHeight = pageHeight - marginTop - marginBottom - SAFETY;
  const titleHeight = titleLineCount * (titlePt * 23 + 200);
  const subtitleHeight = hasSubtitle ? (12 * 23 + 600) : 0;
  const englishLabelHeight = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaHeight = metaLineCount * (10 * 23 + 100);
  const implicitParaHeight = 3 * 300;
  const contentHeight = titleHeight + subtitleHeight + englishLabelHeight + metaHeight + fixedHeight + implicitParaHeight;
  const safeRemaining = Math.max(usableHeight - contentHeight, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(safeRemaining * 0.45);
  const rawBottom = Math.floor(safeRemaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(safeRemaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}

// ---------- R1 封面（design-system.md 配方） ----------
function buildCoverR1(config) {
  const P = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: P.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: P.cover.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.cover.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: P.cover.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: P.cover.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: P.cover.footerColor, font: { ascii: "Arial" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders, children,
      })],
    })],
  })];
}

// ---------- 正文构件 ----------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160, line: 380, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 32, color: "000000",
      font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120, line: 350, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 28, color: "000000",
      font: { eastAsia: "SimHei", ascii: "Times New Roman" } })],
  });
}
function body(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 60 },
    children: [new TextRun({ text, size: 24, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function bodyRuns(runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    indent: { firstLine: 480 },
    spacing: { line: 312, after: 60 },
    children: runs.map(r => new TextRun({ text: r.x, bold: !!r.b, size: 24, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })),
  });
}
function tableTitle(text) {
  return new Paragraph({
    keepNext: true, alignment: AlignmentType.CENTER,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, size: 21, color: "000000",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 80, after: 160 },
    children: [new TextRun({ text, size: 21, color: "404040",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function cellPara(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.CENTER,
    spacing: { line: 280 },
    children: [new TextRun({ text: String(text), bold: !!opts.bold, size: 20,
      color: opts.color || "000000", font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  });
}
function dataTable(spec) {
  const widths = spec.widths;
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: spec.header.map((t, i) => new TableCell({
      children: [cellPara(t, { bold: true, color: PAL.table.headerText })],
      shading: { type: ShadingType.CLEAR, fill: PAL.table.headerBg },
      margins: { top: 60, bottom: 60, left: 100, right: 100 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
    })),
  });
  const rows = spec.rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => new TableCell({
      children: [cellPara(t)],
      shading: ri % 2 === 1 ? { type: ShadingType.CLEAR, fill: PAL.table.surface } : undefined,
      margins: { top: 50, bottom: 50, left: 100, right: 100 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.headerBg },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.headerBg },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: PAL.table.innerLine },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...rows],
  });
}
function image(src, displayWidth, caption) {
  const p = path.join(ROOT, src);
  const buf = fs.readFileSync(p);
  const dim = sizeOf.imageSize ? sizeOf.imageSize(buf) : sizeOf(buf);
  const h = Math.round(displayWidth * dim.height / dim.width);
  const out = [new Paragraph({
    keepNext: true, alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new ImageRun({ data: buf, transformation: { width: displayWidth, height: h }, type: "png" })],
  })];
  if (caption) out.push(figCaption(caption));
  return out;
}

// ---------- 正文组装 ----------
const bodyChildren = [];
for (const b of C.body) {
  if (b.t === "h1") bodyChildren.push(h1(b.x));
  else if (b.t === "h2") bodyChildren.push(h2(b.x));
  else if (b.t === "p") bodyChildren.push(body(b.x));
  else if (b.t === "runs") bodyChildren.push(bodyRuns(b.runs));
  else if (b.t === "table") { bodyChildren.push(tableTitle(b.title)); bodyChildren.push(dataTable(b)); }
  else if (b.t === "img") bodyChildren.push(...image(b.src, b.width, b.caption));
  else if (b.t === "numlist") {
    b.items.forEach((it, i) => bodyChildren.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      indent: { left: 480, hanging: 480 },
      spacing: { line: 312, after: 40 },
      children: [new TextRun({ text: `${i + 1}. ${it}`, size: 24, color: "000000",
        font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
    })));
  } else if (b.t === "ref") {
    bodyChildren.push(new Paragraph({
      alignment: AlignmentType.LEFT,
      indent: { left: 480, hanging: 480 },
      spacing: { line: 312, after: 60 },
      children: [new TextRun({ text: b.x, size: 21, color: "000000",
        font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
    }));
  }
}

// ---------- 前置节：摘要 + 目录 ----------
const frontChildren = [
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 240, after: 300 },
    children: [new TextRun({ text: "摘  要", bold: true, size: 32,
      font: { eastAsia: "SimHei", ascii: "Times New Roman" }, color: "000000" })],
  }),
  ...C.abstract.map(t => body(t)),
  new Paragraph({
    spacing: { before: 200, after: 100 },
    children: [
      new TextRun({ text: "关键词：", bold: true, size: 24, font: { eastAsia: "SimHei", ascii: "Times New Roman" }, color: "000000" }),
      new TextRun({ text: "皮肤病变分类；HAM10000；类别不平衡；迁移学习；卷积神经网络", size: 24, font: { eastAsia: "SimSun", ascii: "Times New Roman" }, color: "000000" }),
      new PageBreak(),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
    children: [new TextRun({ text: "目  录", bold: true, size: 32,
      font: { eastAsia: "SimHei", ascii: "Times New Roman" }, color: "000000" })],
  }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({
    spacing: { before: 200 },
    children: [new TextRun({
      text: "注：本目录由域代码生成。如编辑文档后页码变动，请在目录上右键选择\u201c更新域\u201d以刷新页码。",
      italics: true, size: 18, color: "888888", font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  }),
];

// ---------- 页眉页脚 ----------
const bodyHeader = new Header({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "C0C0C0", space: 4 } },
    children: [new TextRun({ text: "HAM10000 皮肤病变分类实验报告", size: 18, color: "808080",
      font: { eastAsia: "SimSun", ascii: "Times New Roman" } })],
  })],
});
function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })],
    })],
  });
}

// ---------- 文档 ----------
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  creator: "Course Project",
  title: C.cover.title,
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 360, after: 160, line: 380 } },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 240, after: 120, line: 350 } },
      },
    },
  },
  sections: [
    { // 第 1 节：封面（无页码）
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({ ...C.cover, palette: PAL }),
    },
    { // 第 2 节：摘要 + 目录（罗马页码）
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      footers: { default: pageFooter() },
      children: frontChildren,
    },
    { // 第 3 节：正文（阿拉伯页码，从 1 起）
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin,
          pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: bodyHeader },
      footers: { default: pageFooter() },
      children: bodyChildren,
    },
  ],
});

const OUT = path.join(__dirname, "HAM10000实验报告.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
});
