// Thin host boundary. Canonical PPL packages are supplied as unchanged local TGZs.
import { resolveProfileEvent } from '@ppl/profile-runtime';
import { validateProfile, finalizeProfileResolution } from '@ppl/profile-core';
let bytes = 0;
const chunks = [];
try {
  for await (const chunk of process.stdin) {
    bytes += chunk.length;
    if (bytes > 2_000_000) throw new Error('INPUT_TOO_LARGE');
    chunks.push(chunk);
  }
  const input = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  const profile = input.profile;
  if (profile.kind !== 'research' || JSON.stringify(profile.transactionPolicy?.commitEndReasons) !== '["completed"]') {
    throw new Error('PROFILE_POLICY_REJECTED');
  }
  if (validateProfile(profile).length) throw new Error('PROFILE_INVALID');
  const resolution = resolveProfileEvent(profile, input.state ?? profile.initialState, input.event, input.context ?? {});
  const finalized = finalizeProfileResolution(profile, input.state ?? profile.initialState, resolution, input.end_reason);
  const output = {schema_version:'astrograph-ppl-binding-result.v1', request_id:input.request_id,
    committed:finalized.committed, state:finalized.state, diagnostics:resolution.diagnostics,
    active_rules:resolution.activeRules, diagnostic_vote_allowed:false};
  const encoded = JSON.stringify(output);
  if (Buffer.byteLength(encoded) > 2_000_000) throw new Error('OUTPUT_TOO_LARGE');
  process.stdout.write(encoded);
} catch (_) {
  process.stderr.write('PPL_BINDING_FAILED\n');
  process.exitCode = 2;
}
