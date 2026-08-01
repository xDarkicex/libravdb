package LibraVDB::Filter;
use strict;
use warnings;
use JSON::PP;

sub eq {
    my ($class, $field, $value) = @_;
    return { type => 'eq', field => $field, value => $value };
}

sub gt {
    my ($class, $field, $value) = @_;
    return { type => 'gt', field => $field, value => $value };
}

sub lt {
    my ($class, $field, $value) = @_;
    return { type => 'lt', field => $field, value => $value };
}

sub in {
    my ($class, $field, $values) = @_;
    return { type => 'contains_any', field => $field, values => $values };
}

sub and {
    my ($class, @filters) = @_;
    return { type => 'and', filters => \@filters };
}

sub or {
    my ($class, @filters) = @_;
    return { type => 'or', filters => \@filters };
}

sub as_json {
    my ($class, $filter) = @_;
    return encode_json($filter);
}

1;
