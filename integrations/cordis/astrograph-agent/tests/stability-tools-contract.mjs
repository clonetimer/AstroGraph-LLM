import fs from "node:fs";
const source=fs.readFileSync(new URL("../src/stability-tools.ts", import.meta.url),"utf8");
for (const token of ["lineage.step6.assess","release.stability.assess","MAINTAINER","DEVELOPMENT","SHADOW"]) {
  if (!source.includes(token)) throw new Error(`missing ${token}`);
}
if (source.includes('"CONFIRMATORY_SEALED"')) throw new Error("stability mutation tools must not be exposed to sealed confirmatory profile");
console.log("stability Cordis contract PASS");
