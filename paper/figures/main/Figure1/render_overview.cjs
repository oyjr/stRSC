const fs = require('fs');
const path = require('path');
const {chromium} = require('playwright');
(async () => {
  const dir = __dirname;
  const width = 1200, height = 1012;
  const browser = await chromium.launch({headless:true,...(process.env.STRSC_BROWSER ? {executablePath:process.env.STRSC_BROWSER} : {})});
  const page = await browser.newPage({viewport:{width,height},deviceScaleFactor:3});
  const svg = fs.readFileSync(path.join(dir,'Figure1_overview.svg'),'utf8');
  await page.setContent('<html><head><meta charset="utf-8"><style>html,body{margin:0;background:white} @page{size:183mm 154.33mm;margin:0}@media print{body>svg{width:183mm;height:154.33mm}}</style></head><body>'+svg+'</body></html>');
  await page.locator('body > svg').screenshot({path:path.join(dir,'Figure1_overview.png')});
  const labels = await page.locator('svg text').evaluateAll(nodes => nodes.map(n=>{
    const b=n.getBoundingClientRect();
    const style=getComputedStyle(n);
    return {text:n.textContent,x:b.x,y:b.y,right:b.right,bottom:b.bottom,size:parseFloat(style.fontSize),fill:style.fill,opacity:style.opacity,fillOpacity:style.fillOpacity};
  }));
  const outside=labels.filter(b=>b.x<0||b.y<0||b.right>width||b.bottom>height);
  const overlaps=[];
  for(let i=0;i<labels.length;i++)for(let j=i+1;j<labels.length;j++){
    const a=labels[i],b=labels[j];
    const dx=Math.min(a.right,b.right)-Math.max(a.x,b.x),dy=Math.min(a.bottom,b.bottom)-Math.max(a.y,b.y);
    if(dx>1.5&&dy>1.5)overlaps.push({a:a.text,b:b.text,dx,dy});
  }
  const fills=[...new Set(labels.map(t=>t.fill))];
  const allTextOpaque=labels.every(t=>t.opacity==='1'&&t.fillOpacity==='1');
  const arrowCheck = await page.evaluate(() => {
    const boxes = selector => [...document.querySelectorAll(selector)].map(n => n.getBoundingClientRect());
    const textBoxes=boxes('svg text'), points=boxes('svg circle');
    // Use the clipped SVG viewport, not the full bounds of its source image.
    const maps=[...document.querySelectorAll('svg svg')].map(n=>{
      const m=n.parentNode.getScreenCTM();
      const a=new DOMPoint(n.x.baseVal.value,n.y.baseVal.value).matrixTransform(m);
      const b=new DOMPoint(n.x.baseVal.value+n.width.baseVal.value,n.y.baseVal.value+n.height.baseVal.value).matrixTransform(m);
      return {x:Math.min(a.x,b.x),y:Math.min(a.y,b.y),right:Math.max(a.x,b.x),bottom:Math.max(a.y,b.y)};
    });
    const inBox=(p,b,pad=0)=>p.x>=b.x-pad&&p.x<=b.right+pad&&p.y>=b.y-pad&&p.y<=b.bottom+pad;
    const tips=[...document.querySelectorAll('path[stroke="none"][fill="#667781"]')].map(n=>{
      const a=n.getAttribute('d').match(/-?\d*\.?\d+(?:e[-+]?\d+)?/gi).map(Number);
      return new DOMPoint(a[2],a[3]).matrixTransform(n.getScreenCTM());
    });
    return {count:tips.length,textHits:tips.filter(p=>textBoxes.some(b=>inBox(p,b,2))).length,pointHits:tips.filter(p=>points.some(b=>inBox(p,b,1))).length,mapHits:tips.filter(p=>maps.some(b=>inBox(p,b))).length};
  });
  fs.writeFileSync(path.join(dir,'LAYOUT_CHECK.json'),JSON.stringify({outside,textOverlaps:overlaps,minFont:Math.min(...labels.map(t=>t.size)),labelCount:labels.length,textFills:fills,allTextOpaque,arrowCheck},null,2));
  if(arrowCheck.count!==9||arrowCheck.textHits||arrowCheck.pointHits||arrowCheck.mapHits)throw new Error('Arrow landing check failed: '+JSON.stringify(arrowCheck));
  if(fills.length!==1||!allTextOpaque)throw new Error('All text must use one opaque colour');
  if(outside.length)throw new Error('Labels outside art: '+JSON.stringify(outside));
  await page.pdf({path:path.join(dir,'Figure1_overview.pdf'),printBackground:true,preferCSSPageSize:true,margin:{top:0,right:0,bottom:0,left:0},displayHeaderFooter:false});
  await page.setViewportSize({width:1000,height:844});
  await page.addStyleTag({content:'body>svg{width:1000px;height:auto}'});
  await page.locator('body > svg').screenshot({path:path.join(dir,'review_1000.png'),scale:'css'});
  await browser.close();
  console.log(JSON.stringify({png:'3600 x 3036',pdf:'183 x 154.33 mm',textOverlaps:overlaps},null,2));
})();
