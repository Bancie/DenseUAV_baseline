import torch.optim as optim
from torch.optim import lr_scheduler


def make_optimizer(model, opt):
    """Build an SGD optimiser with a multi-step learning-rate scheduler.

    Splits the model parameters into two groups:

    * **Backbone parameters** – identified via ``model.backbone`` – receive a
      reduced learning rate of ``0.3 * opt.lr`` to avoid destroying
      pre-trained representations during fine-tuning.
    * **Head/non-backbone parameters** – all remaining trainable parameters –
      receive the full ``opt.lr``.

    A ``MultiStepLR`` scheduler decays the learning rate by a factor of
    ``0.1`` at epochs 70 and 110.

    Args:
        model (torch.nn.Module): The model to optimise.  Must expose a
            ``model.backbone`` attribute that is a ``torch.nn.Module`` (or
            any object whose ``.parameters()`` can be enumerated).
        opt: Configuration namespace with the following required attributes:

            * ``lr`` (float): Base learning rate for head parameters.

    Returns:
        tuple:
            * **optimizer_ft** (torch.optim.SGD): SGD optimiser with
              Nesterov momentum (``momentum=0.9``) and weight decay
              ``5e-4``.
            * **exp_lr_scheduler** (torch.optim.lr_scheduler.MultiStepLR):
              Learning-rate scheduler that reduces ``lr`` by ``0.1×`` at
              milestones ``[70, 110]``.
    """
    ignored_params = []
    ignored_params += list(map(id, model.backbone.parameters()))
    extra_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    base_params = filter(lambda p: id(p) in ignored_params, model.parameters())
    optimizer_ft = optim.SGD([
        {'params': base_params, 'lr': 0.3 * opt.lr},
        {'params': extra_params, 'lr': opt.lr}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70, 110], gamma=0.1)

    return optimizer_ft, exp_lr_scheduler
