"""OpenPGP formatting with PGPy; Ed25519 signing is delegated to Cloud KMS.

PGPy 0.6's packet adapter is intentionally isolated here. GnuPG interoperability
tests cover certificates, detached signatures and Git annotated tags. No private
OpenPGP key is generated, imported or serialized.
"""

from __future__ import annotations

import copy
from datetime import datetime

from pgpy import PGPKey, PGPUID
from pgpy.constants import ECPointFormat, EllipticCurveOID, HashAlgorithm, KeyFlags, PubKeyAlgorithm
from pgpy.packet.fields import ECPoint, EdDSAPriv, EdDSAPub
from pgpy.packet.packets import PrivKeyV4


class _RemoteMaterial(EdDSAPriv):
    def __init__(self, signer):
        super().__init__()
        self._signer = signer
        self.oid = EllipticCurveOID.Ed25519
        self.p = ECPoint.from_values(256, ECPointFormat.Native, signer.public_key().public_bytes_raw())

    def __privkey__(self):
        # PGPy's EdDSAPriv.sign hashes the OpenPGP packet before calling this.
        # KMS then signs that digest as raw PureEdDSA data, as GnuPG expects.
        return self._signer

    def __bytearray__(self):
        # PGPy uses these public bytes to calculate the fingerprint.
        return EdDSAPub.__bytearray__(self)


class _RemotePacket(PrivKeyV4):
    def __bytearray__(self):
        raise TypeError("Remote OpenPGP private key serialization is forbidden")


def _shell(signer, created: datetime) -> PGPKey:
    key = PGPKey()
    key._key = _RemotePacket()
    key._key.created = created
    key._key.pkalg = PubKeyAlgorithm.EdDSA
    key._key.keymaterial = _RemoteMaterial(signer)
    return key


def create_public_certificate(signer, *, name: str, email: str, created: datetime) -> str:
    if not name or not email or any(c in name + email for c in "\r\n\x00"):
        raise ValueError("A single-line company name and email are required")
    if created.tzinfo is None:
        raise ValueError("OpenPGP creation time must include a timezone")
    key = _shell(signer, created)
    key.add_uid(
        PGPUID.new(name, email=email),
        usage={KeyFlags.Sign, KeyFlags.Certify},
        hashes=[HashAlgorithm.SHA256, HashAlgorithm.SHA512],
        created=created,
    )
    return str(key.pubkey)


def load_signing_certificate(signer, armored: str, *, fingerprint: str) -> PGPKey:
    public, remainder = PGPKey.from_blob(armored)
    if not public.is_public or public.subkeys or public.key_algorithm != PubKeyAlgorithm.EdDSA:
        raise ValueError("Expected a single Ed25519 OpenPGP public primary key")
    if len(remainder) > 1 or str(public.fingerprint) != fingerprint:
        raise ValueError("OpenPGP certificate fingerprint mismatch")
    if public.is_expired or list(public.revocation_signatures):
        raise ValueError("OpenPGP certificate is expired or revoked")
    if public._key.keymaterial.p.x != signer.public_key().public_bytes_raw():
        raise ValueError("OpenPGP certificate and KMS public key differ")
    if not public.userids:
        raise ValueError("OpenPGP certificate has no identity")
    key = _shell(signer, public.created)
    for uid in public.userids:
        if uid.selfsig is None or not public.verify(uid, uid.selfsig):
            raise ValueError("OpenPGP identity self-signature is invalid")
        key |= copy.copy(uid)
    return key


def detached_signature(key: PGPKey, data: bytes) -> str:
    return str(key.sign(data, hash=HashAlgorithm.SHA256))
