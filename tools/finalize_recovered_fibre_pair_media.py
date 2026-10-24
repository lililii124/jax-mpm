"""Render only copy-recovered original full61 states after all original pair gates."""
from pathlib import Path
import json
import finalize_fibre_pair_full61_media as base
from recover_fibre_pair_full61_archive import recover

def restored(model,receipt_sha,root):
 official=json.loads((base.ROOT/'.local/media-official-parents.json').read_text())[model]
 proof=recover(base.ROOT/'.local'/('full-'+model+'.zip'),base.ROOT/'.local'/('full-'+model+'-receipts.zip'),model,root/'copy-only-recovery',official['artifact']['digest'].removeprefix('sha256:'),official['receipts_artifact']['digest'].removeprefix('sha256:'))
 raw=root/'copy-only-recovery/restored/fibre-pull-off-cache';receipt=base.package.read(raw/'receipt.json')
 if proof['original_receipt_sha256']!=receipt_sha:raise ValueError('Exact recovered receipt SHA differs')
 expected=base.package.read(base.ROOT/'tools/fibre_pair_full61_certifications.json')[model]
 if receipt['certified_parent_receipt_sha256']!=expected['receipt_sha256'] or receipt['continued_from_frame']!=2 or receipt['original_steps_not_recomputed']!=974:raise ValueError('Original certified parent lineage differs')
 return raw,receipt,proof
if __name__=='__main__':
 base.restore=restored
 base.main()
