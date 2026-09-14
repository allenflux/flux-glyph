import pytest

torch = pytest.importorskip("torch")
from training.prepare_unified_regions import FAMILIES
from training.short_confidence_retention import CONTRACT, confidence_retention_loss


def batch(specs):
    rows = []
    for target, tile in specs:
        rows.append({"target": target, "family": FAMILIES[target], "tile_count": tile,
                     "split": "train", "native_font_verified": True})
    rows *= 96 // len(rows)
    rows = rows[:96]
    targets = torch.tensor([r["target"] for r in rows], dtype=torch.int64)
    teacher = torch.full((96, 25), -2.0)
    teacher[torch.arange(96), targets] = 2.0
    student = torch.zeros((96, 25), requires_grad=True)
    return rows, targets, student, teacher


def test_short_known_is_boosted_but_multi_known_keeps_teacher_floor():
    rows, targets, student, teacher = batch([(0, 1), (1, 2)])
    loss, eligible, boosted = confidence_retention_loss(student, teacher, targets, rows)
    assert eligible.all() and boosted.tolist() == [True, False] * 48
    assert torch.isfinite(loss) and loss.item() > 0


def test_unknown_is_boosted_and_wrong_teacher_has_no_extra_gradient():
    rows, targets, student, teacher = batch([(24, 2), (2, 1)])
    teacher.fill_(-2.)
    teacher[torch.arange(96), (targets + 1) % 25] = 3.
    loss, eligible, boosted = confidence_retention_loss(student, teacher, targets, rows)
    assert not eligible[0] and not eligible[1] and not boosted.any()
    assert loss.item() == 0.
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_higher_confidence_has_no_penalty_and_teacher_is_detached():
    rows, targets, student, teacher = batch([(0, 1)])
    teacher.requires_grad_(False)
    student.data[torch.arange(96), targets] = 10.
    loss, _, _ = confidence_retention_loss(student, teacher, targets, rows)
    assert loss.item() == 0. and not teacher.requires_grad


def test_extreme_logits_finite_and_contract_is_json_safe():
    rows, targets, _, teacher = batch([(24, 1)])
    student = torch.full((96, 25), -10000., requires_grad=True)
    student.data[:, 24] = 10000.
    loss, _, boosted = confidence_retention_loss(student, teacher, targets, rows)
    assert torch.isfinite(loss) and boosted.all()
    assert CONTRACT["min_probability"] == .8 and CONTRACT["full_KL_preserved"] is True


@pytest.mark.parametrize("fault", ["label", "nontrain", "notnative"])
def test_native_train_identity_is_required(fault):
    rows, targets, student, teacher = batch([(0, 1)])
    if fault == "label": rows[0]["target"] = 1
    elif fault == "nontrain": rows[0]["split"] = "calibration"
    else: rows[0]["native_font_verified"] = False
    with pytest.raises(ValueError):
        confidence_retention_loss(student, teacher, targets, rows)


def test_unknown_correct_teacher_boost_and_multi_known_exact_floor():
    import math
    rows, targets, _, teacher = batch([(24, 2), (1, 2)])
    student=teacher.clone().requires_grad_()
    loss,eligible,boosted=confidence_retention_loss(student,teacher,targets,rows)
    assert eligible.all() and boosted.tolist()==[True,False]*48
    true_logp=teacher.log_softmax(1).gather(1,targets[:,None]).squeeze(1)
    expected=2*(math.log(.8)-true_logp[::2]).sum()/96
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.count_nonzero(student.grad[1::2])==0
    assert (student.grad[::2,24]<0).all()


def test_teacher_above_minimum_is_preserved_without_lowering_floor():
    rows,targets,student,teacher=batch([(0,1)])
    teacher[:,0]=8.
    student.data[:,0]=6.
    loss,eligible,boosted=confidence_retention_loss(student,teacher,targets,rows)
    assert eligible.all() and not boosted.any() and loss>0
    assert torch.allclose(loss,2*(teacher.log_softmax(1)[:,0]-student.log_softmax(1)[:,0]).mean())


def test_tied_or_extreme_wrong_teacher_has_no_extra_gradient():
    rows,targets,student,teacher=batch([(0,1)])
    teacher.zero_()
    teacher[::2,0]=-10000.;teacher[::2,1]=10000.
    loss,eligible,boosted=confidence_retention_loss(student,teacher,targets,rows)
    assert not eligible.any() and not boosted.any() and torch.isfinite(loss) and loss==0
    loss.backward();assert torch.count_nonzero(student.grad)==0


@pytest.mark.parametrize('fault',['teacher_grad','teacher_nan','targets_dtype','targets_value','student_shape','families','tile_count'])
def test_corrupted_teacher_or_contract_is_rejected(fault):
    rows,targets,student,teacher=batch([(0,1)])
    families=list(FAMILIES)
    if fault=='teacher_grad':teacher.requires_grad_()
    elif fault=='teacher_nan':teacher[0,0]=float('nan')
    elif fault=='targets_dtype':targets=targets.float()
    elif fault=='targets_value':targets[0]=1
    elif fault=='student_shape':student=student[:1]
    elif fault=='families':families[0],families[1]=families[1],families[0]
    else:rows[0]['tile_count']=9
    with pytest.raises(ValueError):confidence_retention_loss(student,teacher,targets,rows,families)
