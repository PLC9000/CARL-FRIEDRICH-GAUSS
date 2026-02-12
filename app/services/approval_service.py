"""Approval management: approve / reject recommendations."""

import datetime
import logging

from sqlalchemy.orm import Session

from app.models import Approval, ApprovalStatus, AuditLog

logger = logging.getLogger(__name__)


def _review(
    approval_id: int,
    admin_id: int,
    reason: str,
    new_status: ApprovalStatus,
    db: Session,
) -> Approval:
    """Shared logic for approve / reject."""
    approval = db.query(Approval).filter(Approval.id == approval_id).first()
    if approval is None:
        raise ValueError("Registro de aprobación no encontrado")
    if approval.status != ApprovalStatus.pending:
        raise ValueError(
            f"La aprobación ya está en estado: {approval.status.value}"
        )

    approval.status = new_status
    approval.reviewed_by = admin_id
    approval.review_reason = reason
    approval.reviewed_at = datetime.datetime.utcnow()

    action = "approval.approve" if new_status == ApprovalStatus.approved else "approval.reject"
    db.add(AuditLog(
        user_id=admin_id,
        action=action,
        payload={"approval_id": approval_id, "reason": reason},
    ))
    db.commit()
    db.refresh(approval)
    logger.info("Approval %d %s by admin %d", approval_id, new_status.value, admin_id)
    return approval


def approve(approval_id: int, admin_id: int, reason: str, db: Session) -> Approval:
    return _review(approval_id, admin_id, reason, ApprovalStatus.approved, db)


def reject(approval_id: int, admin_id: int, reason: str, db: Session) -> Approval:
    return _review(approval_id, admin_id, reason, ApprovalStatus.rejected, db)
